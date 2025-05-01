# Contribution Guidelines

Thank you for considering contributing to this project!
Contributions are always welcome and appreciated.

## How to Contribute

Please check the [issue tracker](https://github.com/habedi/cogitator/issues) to see if there is an issue you
would like to work on or if it has already been resolved.

### Reporting Bugs

1. Open an issue on the [issue tracker](https://github.com/habedi/cogitator/issues).
2. Include information such as steps to reproduce, expected/actual behavior, and relevant logs or screenshots.

### Suggesting Features

1. Open an issue on the [issue tracker](https://github.com/habedi/cogitator/issues).
2. Provide details about the feature, its purpose, and potential implementation ideas.

## Submitting Pull Requests

- Ensure all tests pass before submitting a pull request.
- Write a clear description of the changes you made and the reasons behind them.

> [!IMPORTANT]
> It's assumed that by submitting a pull request, you agree to license your contributions under the project's license.

## Development Workflow

### Prerequisites

Install GNU Make on your system if it's not already installed.

```shell
# For Debian-based systems like Debian, Ubuntu, etc.
sudo apt-get install make
```

- Use the `make setup` command to install the development dependencies.

### Code Style

- Use the `make format` command to format the code.

### Running Tests

- Use the `make test` command to run the tests.

### Running Linter Checks

- Use the `make lint` command to run the linter checks.

### See Available Commands

- Run `make help` to see all available commands for managing different tasks.

## Code of Conduct

We adhere to the [Python Software Foundation Code of Conduct](https://policies.python.org/python.org/code-of-conduct).
