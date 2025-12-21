# Cyber-Inference

## Overview

Cyber-Inference is a web gui management tool for running a v1 compatible inference server. Initially focused on llama.cpp inference server, but will be expanded to support other inference servers in the future.  It is designed to be conscious of the systems resources and will dynamically adjust the inference server's resources based on the system's resources.

## Design Requirements

- The WebGUI runs on port 8337 on / by default.
- The /v1/ endpoint is on the same port as the WebGUI.
- The /v1/models endpoint is on the same port as the WebGUI.
- Can run on any platform that supports Python 3.12 and the inference server.
- Supports docker build and run for deployment.
- Memory limits are configurable in the WebGUI, along with model loading and unloading thresholds and time limits.
- LLM endpoints, Embed endpoints, we need to ensure we fill all of the llama.cpp gaps and use all of it's features possible.
- Models must be downloaded to model folder and tracked in the database.

## Features

- [ ] Web GUI for managing the llama.cpp inference server with beautiful CyberPunk theme with Neon Green accents in dark mode, uses responsive design for mobile, tablet, and desktop with dynamic pulsing colors and animations.
- [ ] Automatic installation of the inference server
- [ ] Automatic update of the inference server
- [ ] Dynamic download of the models and model cards from the internet
- [ ] Dynamic creation of the endpoints aligned with the models and model cards
- [ ] V1 endpoint supports streaming and proxies for the inference server hiding dynamically generated endpoints allowing seamless integration for the users.
- [ ] Dynamic creation of the /v1/models endpoint aligned with the models and model cards
- [ ] Dynamic resource allocation
- [ ] Automatic model loading of models after v1 api is called
- [ ] Automatic unloading of models after v1 api is called and model is idle for a period of time (configurable)
- [ ] Monitoring of the inference server
- [ ] Logging of the inference server
- [ ] Error handling
- [ ] Configuration management
- [ ] Admin password protection optionally enabled (admin password is stored in the database)

## Platforms

- [ ] NVIDIA Jetson
- [ ] macOS
- [ ] Linux

## Tech Stack

- [ ] uv
- [ ] Python 3.12
- [ ] FastAPI
- [ ] Flask
- [ ] Custom UI components for the web GUI
- [ ] Docker
- [ ] Llama.cpp (CUDA, Metal, CPU)
- [ ] SQLite3 for state management and configuration
- [ ] HuggingFace API for model card and model download

## Project Structure

- [ ] src/
- [ ] tests/
- [ ] docs/
- [ ] scripts/
- [ ] models/
- [ ] utils/
- [ ] web/
- [ ] data/

## Installation


## ⚖️ License

<p>
GPU GPLv3 Licensed.<p><i> (c)Matthew Rogers 2025. All rights reserved. No Warranty. No Support. No Liability. No Refunds.</p<br>
</i><p>
<em>Free Software</em>
</p>

### Connect With Me 🤝

[![GitHub](https://img.shields.io/badge/GitHub-matthewrogers-181717?style=for-the-badge&logo=github)](https://github.com/matthewrogers)
[![Twitter](https://img.shields.io/badge/Twitter-@matthewrogers-1DA1F2?style=for-the-badge&logo=twitter)](https://x.com/matthewrogers)
[![Website](https://img.shields.io/badge/Web-matthewrogers.org-00ADD8?style=for-the-badge&logo=google-chrome)](https://matthewrogers.org)

![Matthew Rogers](https://github.com/RamboRogers/cyberpamnow/raw/master/media/ramborogers.png)

</div>