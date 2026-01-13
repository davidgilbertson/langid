# This maps Highlight.js names to the equivalent names in "The Stack V2" dataset
hljs_to_stack_map = {
    "c": "C",
    "cpp": "C++",
    "csharp": "C-Sharp",
    "css": "CSS",
    "dart": "Dart",
    "diff": "Diff",
    "go": "Go",
    "graphql": "GraphQL",
    "ini": "INI",
    "java": "Java",
    "javascript": "JavaScript",
    "json": "JSON",
    "kotlin": "Kotlin",
    "less": "Less",
    "lua": "Lua",
    "makefile": "Makefile",
    "xml": "XML",
    "markdown": "Markdown",
    "objectivec": "Objective-C",
    "perl": "Perl",
    "php": "PHP",
    "plaintext": "Text",
    "python": "Python",
    "r": "R",
    "ruby": "Ruby",
    "rust": "Rust",
    "scss": "SCSS",
    "shell": "Shell",
    "sql": "SQL",
    "swift": "Swift",
    "typescript": "TypeScript",
    "vbnet": "Visual_Basic_.NET",
    "wasm": "WebAssembly",
    "yaml": "YAML",
}

# These are detected by medium.com but missing in The Stack V2:
# "bash": None,  # closest is "Shell" (not "bash")
# "php-template": None,  # closest is "HTML+PHP" but not the same thing
# "python-repl": None,

stack_langs = [val for val in hljs_to_stack_map.values() if val]
