<a name="readme-top"></a>

[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/{1:github_username}/{2:repo_name}">
    <img src="assets/googly.png" alt="Logo" width="300" height="200">
  </a>

<h3 align="center">Googly Eyes</h3>

  <p align="center">
    Googly Eyes photo modification web app.
    <br />
    <a href="https://trello.com/b/03moddbM/onfido-project">Project Board</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

The purpose of this project is to provide a humorous and entertaining
experience for users by allowing them to modify photos with googly eyes,
enhancing user engagement and satisfaction with FunnyFaces Inc.'s offerings.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

<!-- GETTING STARTED -->

## Getting Started

### Local Development

#### Prerequisites

- [pyenv](https://github.com/pyenv/pyenv) for managing Python versions.
- [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv) for creating virtual environments.
- [pdm](https://github.com/pdm-project/pdm) for Python dependency management.
- [just cli](https://github.com/casey/just) for task automation.

#### Installation

1. Create Virtual Environment:

```bash
git clone https://github.com/mshaban/googly.git
cd googly
pyenv virtualenv 3.11 googlify
pyenv local googlify
```

2. Install Dependencies:

```bash
# Use -G test for development/test dependencies
pdm install
```

3. Running the app

```bash

## Launch FastAPI & Ray Serve
just run
## Run tests
just test

## run fastapi and ray serve separately
source config/local.env
uvicorn src.app.deployment.fastapi:app --reload --host ${FAST_HOST} --port ${FAST_PORT} &
serve run src.app.deployment.ray:app  --port ${SERVE_PORT} --host ${SERVE_URL}

<!-- USAGE EXAMPLES -->

## Having fun...

# Results are stored in `out` directory
python -m src.runner  "images_dir" # --sample_size 3 // default is all images in directory

```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->

## Roadmap

See the [open issues](https://trello.com/b/03moddbM/onfido-project) for a full list of features and known issues.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->

## Contact

Project Link: [https://github.com/mshaban/googly](https://github.com/mshaban/googly)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

- [openvino is amaaazing](https://github.com/openvinotoolkit/openvino)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

[issues-shield]: https://img.shields.io/github/issues/mshaban/googly.svg?style=for-the-badge
[issues-url]: https://github.com/mshaban/googly/issues
[license-shield]: https://img.shields.io/github/license/mshaban/googly.svg?style=for-the-badge
[license-url]: https://github.com/mshaban/googly/blob/master/LICENSE.txt
