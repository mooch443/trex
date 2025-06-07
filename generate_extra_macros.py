#!/usr/bin/env python3
import sys

def generate_simple_apply_macro(n):
    """
    Generates the definition for SIMPLE_APPLY_n.
    For n==1:
      #define SIMPLE_APPLY_1(m, SEP, p0) m(p0)
    For n>=2:
      #define SIMPLE_APPLY_n(m, SEP, p0, p1, ..., p{n-1}) \
              SIMPLE_APPLY_{n-1}(m, SEP, p0, p1, ..., p{n-2}) SEP() m(p{n-1})
    """
    macro_name = f"SIMPLE_APPLY_{n}"
    # Create parameter list: always "m, SEP, p0, p1, ... p{n-1}"
    params = ["m", "SEP"] + [f"p{i}" for i in range(n)]
    param_list_str = ", ".join(params)
    
    if n == 1:
        expansion = "m(p0)"
    else:
        prev_macro = f"SIMPLE_APPLY_{n-1}"
        prev_params = ["m", "SEP"] + [f"p{i}" for i in range(n-1)]
        prev_params_str = ", ".join(prev_params)
        expansion = f"{prev_macro}({prev_params_str}) SEP() m(p{n-1})"
    
    return f"#define {macro_name}({param_list_str}) \\\n    {expansion}"

def generate_simple_apply_chooser(max_val):
    """
    Generates the SIMPLE_APPLY_CHOOSER macro definition.
    It takes parameters: _1, _2, ..., _{max_val}, NAME, ... and expands to NAME.
    """
    params = [f"_{i}" for i in range(1, max_val + 1)]
    # Join parameters in one line, you may split them across lines if desired.
    params_str = ", ".join(params)
    return f"#define SIMPLE_APPLY_CHOOSER({params_str}, NAME, ...) NAME"

def generate_simple_apply_macro_list(max_val):
    """
    Generates the _SIMPLE_APPLY macro definition.
    It uses SIMPLE_APPLY_CHOOSER to select from a descending list of SIMPLE_APPLY macros.
    """
    macro_names = [f"SIMPLE_APPLY_{i}" for i in range(max_val, 0, -1)]
    macro_names_str = ", ".join(macro_names)
    return (f"#define _SIMPLE_APPLY(m, SEP, ...) SIMPLE_APPLY_CHOOSER(__VA_ARGS__, {macro_names_str})(m, SEP, __VA_ARGS__)")

def generate_evaluate_count(max_val):
    """
    Generates the EVALUATE_COUNT macro.
    It takes parameters: _1, _2, ..., _{max_val}, count, ... and expands to count.
    """
    params = [f"_{i}" for i in range(1, max_val + 1)]
    params_str = ", ".join(params)
    return (f"#define EVALUATE_COUNT( \\\n    {params_str}, count, ...) \\\n    count")

def generate_count_macro(max_val):
    """
    Generates the COUNT macro.
    It expands to: IDENTITY(EVALUATE_COUNT(__VA_ARGS__, <max>, <max-1>, ..., 1))
    """
    numbers = [str(i) for i in range(max_val, 0, -1)]
    numbers_str = ", ".join(numbers)
    return (f"#define COUNT(...) \\\nIDENTITY(EVALUATE_COUNT(__VA_ARGS__, {numbers_str}))")

def generate_sep_macros():
    """
    Generates helper macros for the separator and the main SIMPLE_APPLY macro.
    """
    sep_macros = [
        "#define _SEP_INDIRECT() ,",
        "#define _SEP_EMPTY()",
        "#define SIMPLE_APPLY(m, ...) _SIMPLE_APPLY(m, _SEP_INDIRECT, __VA_ARGS__)"
    ]
    return "\n".join(sep_macros)

def main():
    # Use a single command-line argument for the maximum value (default 50)
    max_val = 50
    if len(sys.argv) > 1:
        try:
            max_val = int(sys.argv[1])
        except ValueError:
            print("Please supply an integer as maximum value.")
            sys.exit(1)
    
    output_lines = []
    
    # Generate SIMPLE_APPLY_1 to SIMPLE_APPLY_max_val
    for i in range(1, max_val + 1):
        output_lines.append(generate_simple_apply_macro(i))
        output_lines.append("")  # Blank line for readability
    
    # Generate SIMPLE_APPLY_CHOOSER
    output_lines.append(generate_simple_apply_chooser(max_val))
    output_lines.append("")
    
    # Generate _SIMPLE_APPLY macro
    output_lines.append(generate_simple_apply_macro_list(max_val))
    output_lines.append("")
    
    # Generate EVALUATE_COUNT macro
    output_lines.append(generate_evaluate_count(max_val))
    output_lines.append("")
    
    # Generate COUNT macro
    output_lines.append(generate_count_macro(max_val))
    output_lines.append("")
    
    # Generate separator macros and the main SIMPLE_APPLY macro
    output_lines.append(generate_sep_macros())
    output_lines.append("")
    
    # Print the full generated source
    print("\n".join(output_lines))

if __name__ == "__main__":
    main()