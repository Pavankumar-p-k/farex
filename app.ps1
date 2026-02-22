param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ArgsFromCaller
)

python scripts/run_all_dashboards.py @ArgsFromCaller
