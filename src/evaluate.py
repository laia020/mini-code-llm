"""Historical evaluation protocol, with bounded subprocess execution.

The AST restrictions are not a general-purpose sandbox.
"""
import ast
import copy
import json
import math
import re
from collections import defaultdict
from pathlib import Path
SAFE_BUILTINS = {'abs': abs, 'all': all, 'any': any, 'bool': bool, 'dict': dict, 'enumerate': enumerate, 'float': float, 'int': int, 'len': len, 'list': list, 'max': max, 'min': min, 'range': range, 'reversed': reversed, 'round': round, 'set': set, 'sorted': sorted, 'str': str, 'sum': sum, 'tuple': tuple, 'zip': zip}
SAFE_METHODS = {'append', 'count', 'get', 'items', 'keys', 'lower', 'upper', 'isalpha', 'isdigit', 'join', 'remove', 'reverse', 'sort', 'strip', 'values'}
FORBIDDEN_NODES = (ast.Import, ast.ImportFrom, ast.ClassDef, ast.AsyncFunctionDef, ast.Await, ast.Global, ast.Nonlocal, ast.With, ast.AsyncWith, ast.Try, ast.Raise, ast.Delete, ast.Yield, ast.YieldFrom, ast.While)

def extract_python_code(text):
    """
    Extrai o primeiro bloco de código Python da resposta da LLM.
    """
    python_block = re.search('```python\\s*(.*?)```', text, re.DOTALL | re.IGNORECASE)
    if python_block:
        return python_block.group(1).strip()
    generic_block = re.search('```\\s*(.*?)```', text, re.DOTALL)
    if generic_block:
        return generic_block.group(1).strip()
    lines = text.strip().splitlines()
    start = next((index for index, line in enumerate(lines) if line.lstrip().startswith('def ')), None)
    if start is None:
        return text.strip()
    candidate = lines[start:]
    for end in range(len(candidate), 0, -1):
        code = '\n'.join(candidate[:end]).strip()
        try:
            ast.parse(code)
            return code
        except SyntaxError:
            continue
    return '\n'.join(candidate).strip()

def parse_python(code):
    try:
        return (ast.parse(code), None)
    except SyntaxError as error:
        return (None, f'SyntaxError: {error.msg}')

def find_function(tree, expected_name):
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == expected_name:
            return node
    return None

def has_correct_signature(tree, expected_name, expected_parameters):
    function = find_function(tree, expected_name)
    if function is None:
        return False
    actual_parameters = [argument.arg for argument in function.args.posonlyargs + function.args.args]
    return actual_parameters == expected_parameters and (not function.args.kwonlyargs) and (function.args.vararg is None) and (function.args.kwarg is None)

def is_safe_to_execute(tree):
    """
    Proteção simples para nosso experimento controlado.

    Não é um sandbox genérico para executar código arbitrário.
    """
    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODES):
            return (False, f'Forbidden node: {type(node).__name__}')
        if isinstance(node, ast.Attribute) and node.attr.startswith('_'):
            return (False, 'Private/dunder attribute access is not allowed')
        if isinstance(node, ast.Name) and node.id.startswith('__'):
            return (False, 'Dunder names are not allowed')
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in SAFE_BUILTINS:
                    return (False, f'Call not allowed: {node.func.id}')
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr not in SAFE_METHODS:
                    return (False, f'Method not allowed: {node.func.attr}')
            else:
                return (False, 'Dynamic call target is not allowed')
    return (True, None)

def values_equal(actual, expected):
    """
    Comparação um pouco mais tolerante para números
    de ponto flutuante.
    """
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(actual, expected, rel_tol=1e-09, abs_tol=1e-09)
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all((values_equal(a, b) for a, b in zip(actual, expected)))
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all((values_equal(actual[key], expected[key]) for key in actual))
    return actual == expected

def _run_contract_tests_inline(code, contract):
    """
    Executa os casos definidos em contracts/families.json.
    """
    namespace = {'__builtins__': SAFE_BUILTINS}
    try:
        exec(code, namespace, namespace)
        function = namespace[contract['family']]
    except Exception as error:
        total = len(contract['tests'])
        return (0, total, [f'{type(error).__name__}: {error}'] * total)
    passed = 0
    details = []
    for case in contract['tests']:
        try:
            arguments = copy.deepcopy(case['args'])
            actual = function(*arguments)
            expected = case['expected']
            type_correct = True
            if 'expected_type' in case:
                type_correct = type(actual).__name__ == case['expected_type']
            value_correct = values_equal(actual, expected)
            test_passed = type_correct and value_correct
            if test_passed:
                passed += 1
                details.append('pass')
            else:
                details.append(f'expected={expected!r}, got={actual!r} ({type(actual).__name__})')
        except Exception as error:
            details.append(f'{type(error).__name__}: {error}')
    return (passed, len(contract['tests']), details)

def percentage(value, total):
    if total == 0:
        return 0.0
    return value / total * 100

def evaluate_records(examples, contracts, input_file="<memory>"):
    total = len(examples)

    valid_python_count = 0
    correct_signature_count = 0
    tasks_passed_count = 0

    unit_tests_passed = 0
    unit_tests_total = 0

    exact_match_count = 0
    code_only_count = 0


    by_family = defaultdict(
        lambda: {
            "examples": 0,
            "tasks_passed": 0,
            "tests_passed": 0,
            "tests_total": 0,
        }
    )


    by_difficulty = defaultdict(
        lambda: {
            "examples": 0,
            "tasks_passed": 0,
            "tests_passed": 0,
            "tests_total": 0,
        }
    )


    evaluated_examples = []


    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    for example in examples:

        family = example["family"]

        contract = contracts[family]

        generated = example["generated"]

        code = extract_python_code(
            generated
        )

        tree, syntax_error = parse_python(
            code
        )

        valid_python = tree is not None

        signature_correct = False
        safe_to_execute = False
        safety_error = None

        tests_passed = 0
        tests_total = len(
            contract["tests"]
        )

        test_details = []


        # -----------------------------------------------------
        # Syntax
        # -----------------------------------------------------

        if valid_python:

            valid_python_count += 1

            # -------------------------------------------------
            # Signature
            # -------------------------------------------------

            signature_correct = (
                has_correct_signature(
                    tree,
                    expected_name=family,
                    expected_parameters=contract[
                        "parameters"
                    ],
                )
            )

            if signature_correct:
                correct_signature_count += 1


            # -------------------------------------------------
            # Safety
            # -------------------------------------------------

            safe_to_execute, safety_error = (
                is_safe_to_execute(tree)
            )


            # -------------------------------------------------
            # Behavioral tests
            # -------------------------------------------------

            if (
                signature_correct
                and safe_to_execute
            ):

                (
                    tests_passed,
                    tests_total,
                    test_details,
                ) = run_contract_tests(
                    code,
                    contract,
                )

            else:

                reason = (
                    safety_error
                    or "Incorrect signature"
                )

                test_details = [
                    reason
                ] * tests_total

        else:

            test_details = [
                syntax_error or "Invalid Python"
            ] * tests_total


        # -----------------------------------------------------
        # Task success
        # -----------------------------------------------------

        task_passed = (
            tests_total > 0
            and tests_passed == tests_total
        )


        # -----------------------------------------------------
        # Exact Match
        # -----------------------------------------------------

        exact_match = (
            code.strip()
            == example["expected_code"].strip()
        )


        # -----------------------------------------------------
        # Did the model output only code?
        # -----------------------------------------------------

        code_only = (
            generated.strip()
            == code.strip()
        )


        if task_passed:
            tasks_passed_count += 1

        unit_tests_passed += tests_passed
        unit_tests_total += tests_total

        if exact_match:
            exact_match_count += 1

        if code_only:
            code_only_count += 1


        # -----------------------------------------------------
        # Statistics by family
        # -----------------------------------------------------

        family_stats = by_family[
            family
        ]

        family_stats["examples"] += 1

        family_stats["tasks_passed"] += int(
            task_passed
        )

        family_stats[
            "tests_passed"
        ] += tests_passed

        family_stats[
            "tests_total"
        ] += tests_total


        # -----------------------------------------------------
        # Statistics by difficulty
        # -----------------------------------------------------

        difficulty = str(
            example["difficulty"]
        )

        difficulty_stats = by_difficulty[
            difficulty
        ]

        difficulty_stats["examples"] += 1

        difficulty_stats[
            "tasks_passed"
        ] += int(task_passed)

        difficulty_stats[
            "tests_passed"
        ] += tests_passed

        difficulty_stats[
            "tests_total"
        ] += tests_total


        # -----------------------------------------------------
        # Save detailed evaluation
        # -----------------------------------------------------

        evaluated_examples.append(
            {
                **example,

                "extracted_code": code,

                "valid_python": valid_python,

                "correct_signature":
                    signature_correct,

                "safe_to_execute":
                    safe_to_execute,

                "exact_match":
                    exact_match,

                "code_only_output":
                    code_only,

                "tests_passed":
                    tests_passed,

                "tests_total":
                    tests_total,

                "task_passed":
                    task_passed,

                "syntax_error":
                    syntax_error,

                "safety_error":
                    safety_error,

                "test_details":
                    test_details,
            }
        )


    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    summary = {

        "input_file": str(input_file),

        "examples": total,

        "valid_python": {
            "count": valid_python_count,
            "percent": percentage(
                valid_python_count,
                total,
            ),
        },

        "correct_signature": {
            "count": correct_signature_count,
            "percent": percentage(
                correct_signature_count,
                total,
            ),
        },

        "tasks_passed": {
            "count": tasks_passed_count,
            "percent": percentage(
                tasks_passed_count,
                total,
            ),
        },

        "unit_tests": {
            "passed": unit_tests_passed,
            "total": unit_tests_total,
            "percent": percentage(
                unit_tests_passed,
                unit_tests_total,
            ),
        },

        "exact_match": {
            "count": exact_match_count,
            "percent": percentage(
                exact_match_count,
                total,
            ),
        },

        "code_only_output": {
            "count": code_only_count,
            "percent": percentage(
                code_only_count,
                total,
            ),
        },

        "by_difficulty": {},

        "by_family": {},
    }


    for difficulty, stats in sorted(
        by_difficulty.items(),
        key=lambda item: int(item[0]),
    ):

        summary["by_difficulty"][
            difficulty
        ] = {
            **stats,

            "task_pass_percent":
                percentage(
                    stats["tasks_passed"],
                    stats["examples"],
                ),

            "unit_test_percent":
                percentage(
                    stats["tests_passed"],
                    stats["tests_total"],
                ),
        }


    for family, stats in sorted(
        by_family.items()
    ):

        summary["by_family"][
            family
        ] = {
            **stats,

            "task_pass_percent":
                percentage(
                    stats["tasks_passed"],
                    stats["examples"],
                ),

            "unit_test_percent":
                percentage(
                    stats["tests_passed"],
                    stats["tests_total"],
                ),
        }


    return evaluated_examples, summary

TIMEOUT = 5.0


def run_contract_tests(code, contract, timeout=None):
    import subprocess
    import sys
    total = len(contract["tests"])
    try:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker"],
            input=json.dumps({"code": code, "contract": contract}),
            text=True, encoding="utf-8", capture_output=True,
            timeout=TIMEOUT if timeout is None else timeout, check=True,
        )
        passed, count, details = json.loads(result.stdout)
        return passed, count, details
    except subprocess.TimeoutExpired:
        return 0, total, ["Timeout: execução excedeu o limite por exemplo"] * total
    except (subprocess.CalledProcessError, ValueError, OSError) as error:
        return 0, total, [f"WorkerError: {type(error).__name__}"] * total


def main(argv=None):
    import argparse
    import sys
    if __package__:
        from .common import ROOT, check_new_outputs, read_jsonl, write_json
    else:
        from common import ROOT, check_new_outputs, read_jsonl, write_json
    if argv is None and sys.argv[1:] == ["--worker"]:
        payload = json.load(sys.stdin)
        tree, error = parse_python(payload["code"])
        safe, reason = is_safe_to_execute(tree) if tree is not None else (False, error)
        if not safe:
            raise ValueError(reason)
        print(json.dumps(_run_contract_tests_inline(payload["code"], payload["contract"])))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--contracts", type=Path, default=ROOT / "contracts/families.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args(argv)
    global TIMEOUT
    if args.timeout <= 0:
        parser.error("--timeout precisa ser positivo")
    TIMEOUT = args.timeout
    try:
        check_new_outputs(args.output, args.summary, inputs=[args.input, args.contracts])
        contracts = {c["family"]: c for c in json.loads(args.contracts.read_text(encoding="utf-8"))["families"]}
        examples = read_jsonl(args.input)
        evaluated, summary = evaluate_records(examples, contracts, args.input)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            for item in evaluated:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        write_json(args.summary, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, KeyError) as error:
        print(f"Falha na avaliação: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
