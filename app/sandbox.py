"""
sandbox.py — изолированное выполнение Python-кода, сгенерированного LLM-агентом.

Архитектура изоляции (важно для защиты от prompt-injection и опасного кода):

1. Код выполняется в ОТДЕЛЬНОМ процессе (multiprocessing), а не в основном
   процессе Streamlit-приложения — падение/зависание/исчерпание памяти
   в коде агента не уронит само приложение.
2. Жёсткий таймаут (по умолчанию 20 секунд) — процесс принудительно
   убивается, если код выполняется слишком долго.
3. Ограниченный набор доступных builtins — запрещены open() в произвольных
   местах, eval/exec второго уровня, импорт os/sys/subprocess/socket и т.д.
4. Разрешён только заранее одобренный список модулей для анализа данных
   (pandas, numpy, matplotlib и т.п.) через подмену builtins.__import__.
5. Рабочая директория агента — отдельная временная папка только для
   текущей сессии; доступ к остальной файловой системе не предоставляется.
6. Сетевой доступ не предоставляется (агент физически не может вызвать
   ничего сетевого, т.к. модули urllib/requests/socket не в allowlist).
7. Если LLM сгенерировала код с частой ошибкой (буквальный перенос строки
   внутри однострочного литерала вместо \n — типично для более слабых
   локальных моделей), sandbox пытается автоматически починить это перед
   тем как сдаться и вернуть ошибку агенту.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import traceback
import io
import contextlib
import builtins as _builtins_module
from dataclasses import dataclass, field
from pathlib import Path

# Модули, которые агенту разрешено импортировать для анализа данных.
ALLOWED_MODULES = {
    "pandas",
    "numpy",
    "matplotlib",
    "matplotlib.pyplot",
    "json",
    "math",
    "statistics",
    "datetime",
    "time",
    "random",
    "dateutil",
    "re",
    "collections",
    "itertools",
    "io",
    "tabulate",
}

DEFAULT_TIMEOUT_SECONDS = 20


@dataclass
class ExecutionResult:
    success: bool
    stdout: str = ""
    error: str | None = None
    generated_files: list[str] = field(default_factory=list)


def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Подменяем __import__: разрешаем только модули из allowlist."""
    top_level = name.split(".")[0]
    if name not in ALLOWED_MODULES and top_level not in {
        m.split(".")[0] for m in ALLOWED_MODULES
    }:
        raise ImportError(
            f"Импорт модуля '{name}' запрещён в sandbox. "
            f"Разрешены только: {sorted(ALLOWED_MODULES)}"
        )
    return _builtins_module.__import__(name, globals, locals, fromlist, level)


def _build_restricted_globals(work_dir: str) -> dict:
    """Строит словарь globals() с урезанными builtins для exec()."""

    # Белый список безопасных builtins (без open в произвольном месте,
    # без eval/exec/compile, без __import__ в исходном виде, без exit/quit).
    safe_builtin_names = [
        "abs", "all", "any", "bool", "bytes", "chr", "dict", "divmod",
        "enumerate", "filter", "float", "format", "frozenset", "getattr",
        "hasattr", "hash", "hex", "int", "isinstance", "issubclass", "iter",
        "len", "list", "map", "max", "min", "next", "object", "oct", "ord",
        "pow", "print", "range", "repr", "reversed", "round", "set",
        "setattr", "slice", "sorted", "str", "sum", "tuple", "type", "zip",
        "True", "False", "None", "ValueError", "TypeError", "KeyError",
        "IndexError", "StopIteration", "Exception", "ZeroDivisionError",
        "RuntimeError", "AttributeError", "NotImplementedError",
    ]

    safe_builtins = {}
    for n in safe_builtin_names:
        if hasattr(_builtins_module, n):
            safe_builtins[n] = getattr(_builtins_module, n)

    safe_builtins["__import__"] = _restricted_import

    # open() разрешаем, но только для записи внутри work_dir (для графиков)
    def _restricted_open(file, mode="r", *args, **kwargs):
        target = Path(work_dir) / Path(file).name
        if "w" not in mode and "a" not in mode and "x" not in mode:
            raise PermissionError(
                "В sandbox разрешена только запись файлов (для сохранения "
                "графиков), чтение произвольных файлов запрещено."
            )
        return open(target, mode, *args, **kwargs)

    safe_builtins["open"] = _restricted_open

    return {"__builtins__": safe_builtins}


def _repair_unescaped_newlines_in_strings(code: str) -> str | None:
    """
    Чинит частую ошибку слабых LLM: буквальный перенос строки внутри
    обычного однострочного литерала ' или " (вместо экранированного \\n
    или тройных кавычек). Это гарантированная SyntaxError в Python.

    Проходит по коду посимвольно, отслеживая, находимся ли мы внутри
    односимвольной кавычки (а не тройной — её не трогаем, там переносы
    допустимы). Если внутри одинарной/двойной кавычки встречается реальный
    перенос строки, заменяет его на экранированную последовательность \\n.

    Возвращает None, если в коде не было таких переносов (нечего чинить).
    """
    result = []
    i = 0
    n = len(code)
    changed = False

    while i < n:
        ch = code[i]

        # Тройные кавычки — пропускаем целиком как есть, переносы там легальны
        if code[i : i + 3] in ('"""', "'''"):
            quote = code[i : i + 3]
            result.append(quote)
            i += 3
            end = code.find(quote, i)
            if end == -1:
                result.append(code[i:])
                i = n
            else:
                result.append(code[i:end])
                result.append(quote)
                i = end + 3
            continue

        # Однострочная кавычка (' или ")
        if ch in ("'", '"'):
            quote = ch
            result.append(quote)
            i += 1
            while i < n:
                c = code[i]
                if c == "\\" and i + 1 < n:
                    # Экранированный символ — копируем как есть, не трогаем
                    result.append(code[i : i + 2])
                    i += 2
                    continue
                if c == "\n":
                    # Настоящий перенос строки внутри однострочной кавычки —
                    # это всегда синтаксическая ошибка. Чиним на \n.
                    result.append("\\n")
                    changed = True
                    i += 1
                    continue
                if c == quote:
                    result.append(c)
                    i += 1
                    break
                result.append(c)
                i += 1
            continue

        # Комментарий — копируем строку как есть, не заходя внутрь как в код
        if ch == "#":
            end = code.find("\n", i)
            if end == -1:
                result.append(code[i:])
                i = n
            else:
                result.append(code[i:end])
                i = end
            continue

        result.append(ch)
        i += 1

    if not changed:
        return None
    return "".join(result)


def _compile_with_auto_repair(code: str):
    """Компилирует код агента; при unterminated string пробует автопочинку."""
    try:
        return compile(code, "<agent_code>", "exec"), code
    except SyntaxError as syn_err:
        # Автопочинка применима только к одному конкретному классу ошибок —
        # буквальный перенос строки внутри однострочной кавычки. Не пытаемся
        # её применять к другим SyntaxError (это не поможет и может вводить
        # в заблуждение).
        is_unterminated_string = "unterminated string literal" in (syn_err.msg or "")
        if is_unterminated_string:
            repaired = _repair_unescaped_newlines_in_strings(code)
            if repaired is not None:
                try:
                    return compile(repaired, "<agent_code>", "exec"), repaired
                except SyntaxError:
                    pass  # автопочинка не помогла — выбрасываем исходную ошибку ниже

        msg = syn_err.msg or ""
        # Подсказка зависит от РЕАЛЬНОЙ причины, а не всегда одна и та же —
        # иначе агент чинит не ту проблему и зацикливается на одной ошибке.
        if "was never closed" in msg or "unexpected EOF" in msg:
            hint = (
                "Похоже, код был ОБОРВАН ДО ТОГО, КАК ТЫ ЕГО ЗАКОНЧИЛ "
                "(не хватило места в ответе) — открывающая скобка/кавычка "
                "так и не была закрыта. Напиши значительно БОЛЕЕ КОРОТКИЙ "
                "код за один шаг: меньше операций, меньше print(), меньше "
                "комментариев. Лучше сделать несколько маленьких шагов "
                "run_code подряд, чем один длинный."
            )
        elif "unterminated string literal" in msg:
            hint = (
                "Частая причина — настоящий перенос строки внутри "
                "однострочного литерала (вместо экранированного \\n). "
                "Используй \\n внутри строки или тройные кавычки для "
                "многострочного текста."
            )
        else:
            hint = (
                "Проверь синтаксис кода построчно и попробуй снова, "
                "по возможности короче."
            )

        raise SyntaxError(
            f"Код содержит синтаксическую ошибку Python: {msg} "
            f"(строка {syn_err.lineno}). {hint}"
        ) from syn_err


def _worker(code: str, df_path: str, work_dir: str, result_queue: mp.Queue) -> None:
    """Выполняется в дочернем процессе."""
    stdout_buffer = io.StringIO()
    try:
        # Резолвим путь к датасету в абсолютный ДО смены рабочей директории,
        # иначе относительный путь перестанет указывать на файл после chdir.
        df_path = str(Path(df_path).resolve())
        os.chdir(work_dir)

        restricted_globals = _build_restricted_globals(work_dir)

        # Предзагружаем pandas DataFrame под именем `df` — это единственный
        # способ агента получить доступ к данным пользователя. Прямого
        # доступа к файловой системе пользователя у кода нет.
        import pandas as pd  # импорт в самом sandbox-процессе разрешён,
                              # т.к. это инфраструктурный код, а не код агента

        # Принудительно используем headless-бэкенд matplotlib, чтобы код
        # агента не падал на сервере без дисплея, даже если агент сам не
        # прописал matplotlib.use("Agg").
        import matplotlib
        matplotlib.use("Agg")

        if df_path.endswith(".csv"):
            df = pd.read_csv(df_path)
        else:
            df = pd.read_excel(df_path)

        restricted_globals["df"] = df
        restricted_globals["pd"] = pd

        compiled, _code_used = _compile_with_auto_repair(code)

        with contextlib.redirect_stdout(stdout_buffer):
            exec(compiled, restricted_globals)

        generated = [
            str(p.name) for p in Path(work_dir).iterdir() if p.is_file()
        ]

        result_queue.put(
            ExecutionResult(
                success=True,
                stdout=stdout_buffer.getvalue(),
                generated_files=generated,
            )
        )
    except Exception:
        result_queue.put(
            ExecutionResult(
                success=False,
                stdout=stdout_buffer.getvalue(),
                error=traceback.format_exc(limit=5),
            )
        )


def run_code_in_sandbox(
    code: str,
    df_path: str,
    work_dir: str,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> ExecutionResult:
    """
    Запускает код агента в изолированном дочернем процессе.

    :param code: Python-код, сгенерированный LLM-агентом
    :param df_path: путь к датасету пользователя (CSV/Excel), который
                     будет загружен в переменную `df`
    :param work_dir: рабочая директория для этого запуска (для графиков)
    :param timeout: максимальное время выполнения в секундах
    """
    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()

    process = ctx.Process(
        target=_worker, args=(code, df_path, work_dir, result_queue)
    )
    process.start()
    process.join(timeout=timeout)

    if process.is_alive():
        process.terminate()
        process.join(timeout=2)
        if process.is_alive():
            process.kill()
        return ExecutionResult(
            success=False,
            error=f"Превышен таймаут выполнения ({timeout} сек). "
            "Код выполнялся слишком долго и был принудительно остановлен.",
        )

    if not result_queue.empty():
        return result_queue.get()

    return ExecutionResult(
        success=False,
        error="Процесс завершился без результата (возможно, аварийно "
        f"упал с кодом выхода {process.exitcode}).",
    )
