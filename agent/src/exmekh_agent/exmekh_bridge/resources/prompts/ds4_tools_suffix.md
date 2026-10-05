
</tools>

If you choose to call a function, ONLY reply in the following DSML format with NO suffix:

<｜DSML｜tool_calls>
<｜DSML｜invoke name="$TOOL_NAME">
<｜DSML｜parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>

<IMPORTANT>
Reminder:
- Function calls MUST use the DSML format above; native <tool_call> syntax is not executable in this runtime
- Required parameters MUST be specified with their exact schema names
- String parameters use raw text and string="true"; numbers, booleans, arrays, and objects use JSON and string="false"
- When an instruction requires a listed function, call it instead of answering with prose
- If no function call is needed, answer normally and do not mention function calls
</IMPORTANT>