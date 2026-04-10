#!/usr/bin/env python3
import os
import re

def process_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if not lines:
            return None

        # 1. Find the shebang or the first meaningful line
        # We need to remove anything before #!/usr/bin/env python3
        shebang_pattern = re.compile(r'^#!/usr/bin/env python3')
        
        start_index = 0
        found_shebang = False
        for i, line in enumerate(lines):
            if shebang_pattern.match(line.strip()):
                start_index = i
                found_shebang = True
                break
        
        # If no shebang found, we'll prepend it later or just ensure the first line is clean.
        # But the requirement says: "Remove any text before the #! line. Do not insert text before the #! line."
        # This implies if there IS a #! line, we strip everything before it.
        
        new_lines = []
        if found_shebang:
            new_lines = lines[start_index:]
        else:
            # If no shebang found, the user instruction is slightly ambiguous.
            # "Remove any text before the #! line." 
            # If it doesn't exist, we can't remove text *before* it.
            # However, the user says: "#!/usr/bin/env python3 should be at the top of every python program."
            # So if it's missing, we should probably add it at the very top.
            new_lines = ["#!/usr/bin/env python3\n"] + lines

        # 2. Check for Module Docstring
        # A simple check: is there a docstring at the top after the shebang?
        # We'll look for the first non-empty, non-comment line after shebang.
        
        has_module_docstring = False
        content_index = 0
        for i, line in enumerate(new_lines):
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                content_index = i
                break
        
        # If the next meaningful content isn't a docstring, we might need to add one.
        # For this automated task, we'll focus on the 'shebang' part as it's high risk/high reward.
        # Adding docstrings is much more complex and requires understanding the code.
        # I will implement the shebang cleanup first.

        if len(new_lines) != len(lines) or (found_shebang and start_index > 0):
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            return f"Fixed shebang in {file_path}"
        
        return None

    except Exception as e:
        return f"Error processing {file_path}: {e}"

def main():
    with open('python_files_to_document.txt', 'r') as f:
        files = [line.strip() for line in f if line.strip()]

    results = []
    for file_path in files:
        res = process_file(file_path)
        if res:
            results.append(res)
    
    if results:
        print("\n".join(results))
    else:
        print("No changes needed.")

if __name__ == "__main__":
    main()
