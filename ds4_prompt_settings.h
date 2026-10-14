#ifndef DS4_PROMPT_SETTINGS_H
#define DS4_PROMPT_SETTINGS_H

/* Product text owned by the server. Macros preserve literal concatenation. */
#define DS4_PROMPT_STRUCTURED_OUTPUT_GUIDANCE \
    "If thinking is enabled, keep all reasoning (including JSON examples) " \
    "before the closing </think> tag. Immediately after </think>, return " \
    "exactly one JSON value and no prose, Markdown, or code fences. " \
    "The JSON value must satisfy this JSON Schema exactly:\n"
#define DS4_PROMPT_TOOL_SEARCH_DESCRIPTION "Search available tools."
#define DS4_PROMPT_DEEPSEEK_TOOLS_INSTRUCTION \
    "## Tools\n\n" \
    "You have access to a set of tools to help answer the user question. " \
    "You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like the following:\n\n" \
    "<｜DSML｜tool_calls>\n" \
    "<｜DSML｜invoke name=\"$TOOL_NAME\">\n" \
    "<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n" \
    "...\n" \
    "</｜DSML｜invoke>\n" \
    "<｜DSML｜invoke name=\"$TOOL_NAME2\">\n" \
    "...\n" \
    "</｜DSML｜invoke>\n" \
    "</｜DSML｜tool_calls>\n\n" \
    "String parameters should be specified as raw text and set `string=\"true\"`. " \
    "Preserve characters such as `>`, `&`, and `&&` exactly; never replace normal string characters with XML or HTML entity escapes. " \
    "Only if a string value itself contains the exact closing parameter tag `</｜DSML｜parameter>`, write that tag as `&lt;/｜DSML｜parameter>` inside the value. " \
    "For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\n" \
    "When thinking mode is enabled, finish reasoning with </think> before any tool calls or final response.\n\n" \
    "Otherwise, output directly after </think> with tool calls or final response.\n\n" \
    "### Available Tool Schemas\n\n"
#define DS4_PROMPT_TOOL_SCHEMA_GUIDANCE \
    "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls. " \
    "Use the exact parameter names from the schemas."
#define DS4_PROMPT_QWEN_TOOLS_PREFIX \
    "# Tools\n\n" \
    "You have access to the following functions:\n\n<tools>\n"
#define DS4_PROMPT_QWEN_TOOLS_SUFFIX \
    "\n</tools>\n\n" \
    "If you choose to call a function, ONLY reply in the following DSML " \
    "format with NO suffix:\n\n" \
    "<｜DSML｜tool_calls>\n" \
    "<｜DSML｜invoke name=\"$TOOL_NAME\">\n" \
    "<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n" \
    "</｜DSML｜invoke>\n" \
    "</｜DSML｜tool_calls>\n\n" \
    "<IMPORTANT>\n" \
    "Reminder:\n" \
    "- Function calls MUST use the DSML format above; native <tool_call> " \
    "syntax is not executable in this runtime\n" \
    "- Required parameters MUST be specified with their exact schema names\n" \
    "- String parameters use raw text and string=\"true\"; numbers, booleans, " \
    "arrays, and objects use JSON and string=\"false\"\n" \
    "- When an instruction requires a listed function, call it instead of " \
    "answering with prose\n" \
    "- If no function call is needed, answer normally and do not mention " \
    "function calls\n" \
    "</IMPORTANT>"
#define DS4_PROMPT_GLM_TOOLS_PREFIX \
    "\n# Tools\n\n" \
    "You may call one or more functions to assist with the user query.\n\n" \
    "You are provided with function signatures within <tools></tools> XML tags:\n" \
    "<tools>\n\n"
#define DS4_PROMPT_TOOL_SCHEMA_SEPARATOR "\n\n\n"
#define DS4_PROMPT_GLM_TOOLS_SUFFIX \
    "</tools>\n\n" \
    "For each function call, output the function name and arguments within the following XML format:\n" \
    "<tool_call>{function-name}" \
    "<arg_key>{arg-key-1}</arg_key>" \
    "<arg_value>{arg-value-1}</arg_value>" \
    "<arg_key>{arg-key-2}</arg_key>" \
    "<arg_value>{arg-value-2}</arg_value>" \
    "...</tool_call>"

#define DS4_PROMPT_DEEPSEEK41_TOOLS \
    "## Tools\n\n" \
    "You can invoke tools using this format:\n\n" \
    DS41_TOOL_CALLS_START "\n" \
    DS41_INVOKE_START " name=\"$TOOL_NAME\">\n" \
    DS41_PARAM_START " name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE" DS41_PARAM_END "\n" \
    DS41_INVOKE_END "\n" DS41_TOOL_CALLS_END "\n\n" \
    "String values use string=\"true\"; all other values use JSON and string=\"false\". " \
    "Inside string values only, escape a literal </｜DSML｜ parameter> as &lt;/｜DSML｜ parameter>. " \
    "To write that escaped spelling literally, use &amp;lt;/｜DSML｜ parameter>. Other HTML entities are unchanged.\n\n" \
    "Finish reasoning with </think> before tool calls or a final response.\n\n" \
    "### Available Tool Schemas\n\n"

#define DS4_PROMPT_REASONING_MAX_LABEL "Reasoning Effort: Max"

#define DS4_PROMPT_QWEN4_TOOLS_PREFIX "# Tools\n\nYou have access to the following functions:\n\n<tools>"

#define DS4_PROMPT_QWEN4_TOOLS_SUFFIX \
    "\n</tools>\n\n" \
    "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n" \
    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n" \
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n" \
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n" \
    "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested " \
    "within <tool_call></tool_call> XML tags\n" \
    "- Required parameters MUST be specified\n" \
    "- You may provide optional reasoning for your function call in natural language BEFORE the function call, " \
    "but NOT after\n" \
    "- If there is no function call available, answer the question like normal with your current knowledge and " \
    "do not tell the user about function calls\n</IMPORTANT>"

#define DS4_PROMPT_DEFAULT_SYSTEM_MESSAGE "You are a helpful assistant"

#define DS4_PROMPT_INVALID_DSML_TOOL_GUIDANCE \
    "\nThe previous assistant output was not executed because the DSML syntax was malformed. " \
    "Emit a new valid DSML tool call, or answer normally if no tool is needed."

#define DS4_PROMPT_SYSTEM_REMINDER "\n\nSystem prompt reminder:\n"

#define DS4_PROMPT_INVALID_GLM_TOOL_GUIDANCE \
    "\nThe previous assistant output was not executed because the " \
    "<tool_call> syntax was malformed. Emit a new valid <tool_call>, " \
    "or answer normally if no tool is needed."

#define DS4_PROMPT_INVALID_QWEN_TOOL_GUIDANCE \
    "\nThe previous assistant output was not executed because the " \
    "<tool_call><function=...> syntax was malformed. Emit a new valid " \
    "<tool_call>, or answer normally if no tool is needed."

#endif
