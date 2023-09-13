for /r %%f in (*.py) do (
    pyupgrade --py38-plus "%%f"
)
autoflake --remove-all-unused-imports --remove-unused-variables --remove-duplicate-keys --in-place --recursive .
isort .
black .
coverage run -m unittest discover tests
coverage xml
