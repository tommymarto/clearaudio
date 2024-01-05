# Clear Audio

## Installation
Uses poetry as a package manager. Only requires the standard pyproject.toml file.

Clone the project, install poetry (if not installed)

```
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```

Then simply run in the main folder:

```
poetry install
```

## Run
First launch the virtual env with:

`poetry shell`

Then use the following commands

```
poe create dataset=<thedataset>
```

Running, the following command will show available datasets 
```
poe create
``` 

## Additional datasets
If you want to override a setting or add new dataset, you can add create a new config file in the `additional_conf/dataset` folder. Check the `extra_dataset` example.

You can also override the search path completely with the following command
```
python create_dataset.py 'hydra.searchpath=[pkg://relative_path_to_folder]'
```

## SCITAS

https://scitas-data.epfl.ch/confluence/display/DOC/Python+Virtual+Environments