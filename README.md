<a id="readme-top"></a>

<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![project_license][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/mariocosenza/kgsum">
    <img src="images/logo.png" alt="Logo" width="100" height="80">
  </a>

<h3 align="center">KgSum</h3>

  <p align="center">
    A Python application for extracting, preparing, and classifying Knowledge Graphs, leveraging LLMs and traditional machine learning.<br>
    <b>Thesis Project, University of Salerno, ISISLab</b>
    <br />
    <a href="https://github.com/mariocosenza/kgsum/wiki"><strong>Explore the docs »</strong></a>
    <br />
    <a href="https://github.com/mariocosenza/kgsum/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/mariocosenza/kgsum/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
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
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://github.com/mariocosenza/kgsum)

**KgSum** is a Python application for extracting, preparing, and classifying Knowledge Graphs (KGs). It combines Large Language Models (such as Mistral Instructor 7B with QLoRA) and traditional machine learning for effective graph classification and profiling.

Thesis Project for Bachelor's Degree  
University of Salerno  
Lab: ISISLab  
Author: Mario Cosenza  
Supervisor: Maria Angela Pellegrino  

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* [Python](https://www.python.org/)
* [CUDA](https://developer.nvidia.com/cuda-toolkit)
* NVIDIA GPUs

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

Follow these steps to set up KgSum locally.

### Prerequisites

- Python 3.12
- CUDA 12.8
- NVIDIA GPU (recommended: RTX 3070 or higher)

Install dependencies:
```sh
pip install -r requirements.txt
```

### Installation

1. Set required environment variables:
    - `GEMINI_API_KEY`: API key for Gemini models, required for data extraction
    - `LOCAL_ENDPOINT_LOV`: URL of the local SPARQL endpoint for LOV

2. Clone the repository:
   ```sh
   git clone https://github.com/mariocosenza/kgsum.git
   cd kgsum
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

### Data Extraction Workflow

1. Download the latest JSON snapshot from the LOD Cloud.
2. Set `GEMINI_API_KEY` in your environment.
3. Run the scripts in order:
   ```sh
   python endpoint_lod_service.py
   python github_search.py
   python zenodo_records_extraction.py
   ```
   **Note:** Ensure you do not exceed the API rate limits.

### Data Preparation

Prepare local datasets:

```sh
python data_preparation.py
python data_preparation_remote.py
```

(Optional) Include LOV tags and comments:

```sh
python lov_data_preparation.py
```

Make sure `LOCAL_ENDPOINT_LOV` points to your SPARQL endpoint.

### Utility & Preprocessing

```sh
python util.py
python preprocessing.py
```

### Pipeline Build & Training

1. Open `pipeline_build.py`
2. Specify in the code:
    - Classifier type (LLM or traditional ML)
    - Features to use
3. Execute:
   ```sh
   python pipeline_build.py
   ```

### Running the Application

Start the web service with the trained model:

```sh
python app.py
```

### API Usage

Send POST requests to:
- `/api/v1/profile/sparql`
- `/api/v1/profile/file`

Refer to the Swagger documentation (coming soon) for request and response formats.

### Hardware Requirements

| Component | Minimum Specification                           |
|-----------|------------------------------------------------|
| GPU       | NVIDIA GPU with 8+ GB VRAM (RTX 3070 or equiv) |
| RAM       | 32 GB                                          |
| CPU       | Modern multi-core processor                    |
| CUDA      | 12.8                                           |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [ ] Add Swagger API documentation
- [ ] Expand coverage for more LLMs
- [ ] Add more dataset preparation examples
- [ ] Add Docker support

See the [open issues](https://github.com/mariocosenza/kgsum/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Top contributors:

<a href="https://github.com/mariocosenza/kgsum/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=mariocosenza/kgsum" alt="contrib.rocks image" />
</a>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Mario Cosenza - [@mario_cosenza_](https://x.com/mario_cosenza_) - cosenzamario@proton.me  
Supervisor: Maria Angela Pellegrino

Project Link: [https://github.com/mariocosenza/kgsum](https://github.com/mariocosenza/kgsum)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* University of Salerno, ISISLab
* [Mistral LLM](https://mistral.ai/)
* [LOD Cloud](https://lod-cloud.net/)
* [Zenodo](https://zenodo.org/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/mariocosenza/kgsum.svg?style=for-the-badge
[contributors-url]: https://github.com/mariocosenza/kgsum/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/mariocosenza/kgsum.svg?style=for-the-badge
[forks-url]: https://github.com/mariocosenza/kgsum/network/members
[stars-shield]: https://img.shields.io/github/stars/mariocosenza/kgsum.svg?style=for-the-badge
[stars-url]: https://github.com/mariocosenza/kgsum/stargazers
[issues-shield]: https://img.shields.io/github/issues/mariocosenza/kgsum.svg?style=for-the-badge
[issues-url]: https://github.com/mariocosenza/kgsum/issues
[license-shield]: https://img.shields.io/github/license/mariocosenza/kgsum.svg?style=for-the-badge
[license-url]: https://github.com/mariocosenza/kgsum/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/mariocosenza
[product-screenshot]: images/logo_isis.png