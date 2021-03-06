[![release version](https://img.shields.io/badge/dynamic/json?url=https://raw.githubusercontent.com/gdadunashvili/diffconvolver/main/VERSION.json&query=$.*&color=blue&label=version)](https://github.com/gdadunashvili/diffconvolver/releases)
[![licence](https://img.shields.io/badge/licence-MIT-green)](https://github.com/gdadunashvili/diffconvolver/blob/main/LICENSE)

# difconvolver

This project originated as a way to solve a specific coupled PDE system.

It implements several classes that make setting up an FDM script easy like:
A class to generate a 
- 2D Grid
- spacial derivative operators
- bulk updaters
- general updaters for double-periodic b.c.'s

The project is not really general as is but is easily generalisable to different PDE aystems

## Installation

Easiest way to install this package is to use `pip`
```
pip install git+https://github.com/gdadunashvili/diffconvolver/        
```
