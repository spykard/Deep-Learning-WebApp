# Deploy a Deep Learning Model as a Web App in 10 Minutes

[![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](http://perso.crans.org/besson/LICENSE.html)
![Python Version](https://img.shields.io/badge/python-3.6%2B-green.svg)
![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)

A pretty and customizable web app to deploy your Deep Learning Models with ease

## Getting Started

- Clone this repo
- Install requirements
- Run the script
- Go to http://localhost:5000
- Done! :tada:

<p align="center">
  <img src="https://github.com/spykard/Deep-Learning-WebApp/blob/master/screenshots/demo_PC.png?raw=true" height="560px" alt="Example of the Implementation in action">
</p>

## New Features :fire:

- Enhanced, mobile-friendly UI
- Support image drag-and-drop
- State-of-the-art custom-made text preprocessing
- Use vanilla JavaScript, HTML and CSS. Remove jQuery and Bootstrap
- Upgrade Docker base image to Python 3

<p float="left">
  <img src="https://github.com/spykard/Deep-Learning-WebApp/blob/master/screenshots/demo_tablet.png?raw=true" height="330px" alt="">
  <img src="https://github.com/spykard/Deep-Learning-WebApp/blob/master/screenshots/demo_phone.png?raw=true" height="330px" alt="">
</p>

------------------

## Run with Docker

With **[Docker](https://www.docker.com)**, you can quickly build and run the entire application in minutes :whale:

```shell
# 1. First, clone the repo
$ git clone https://github.com/spykard/Deep-Learning-WebApp.git
$ cd Deep-Learning-WebApp

# 2. Build Docker image
$ docker build -t keras_flask_app .

# 3. Run!
$ docker run -it --rm -p 5000:5000 keras_flask_app
```

Open http://localhost:5000 and wait till the webpage is loaded.

## Local Installation

It's easy to install and run the app on your computer.

```shell
# 1. First, clone the repo
$ git clone https://github.com/spykard/Deep-Learning-WebApp.git
$ cd Deep-Learning-WebApp

# 2. Install Python packages
$ pip install -r requirements.txt

# 3. Run!
$ python app.py
```

Open http://localhost:5000 and have fun. :smiley:

------------------

## Customization

It's also easy to customize and include your own models in this app.

<details>
 <summary>Details</summary>

### Use your own model

Place your trained `.h5` file saved by `model.save()` under the models directory.

Change the [code in app.py](https://github.com/spykard/Deep-Learning-WebApp/blob/master/app.py#L25) and make the appropriate changes in the preprocessing modules ([deeplearning_image.py](https://github.com/spykard/Deep-Learning-WebApp/blob/master/deeplearning_image.py) and [deeplearning_text.py](https://github.com/spykard/Deep-Learning-WebApp/blob/master/deeplearning_text.py)) to fit your model's needs.

### Use other pre-trained model

See [Keras applications](https://keras.io/applications/) for more available models, such as DenseNet, MobilNet, NASNet, etc.

Check [this section in app.py](https://github.com/spykard/Deep-Learning-WebApp/blob/master/app.py#L20).

### UI Modification

Modify files in `templates` and `static` directory.

`index.html` implements the UI and `main.js` implements all the behaviors.

</details>


## Deployment

To deploy it for public use, you need to have a public **linux server**.

<details>
 <summary>Details</summary>
  
### Run the app

Run the script and hide it in background with `tmux` or `screen`.
```
$ python app.py
```

You can also use gunicorn instead of gevent
```
$ gunicorn -b 127.0.0.1:5000 app:app
```

For more deployment options, check [here](https://flask.palletsprojects.com/en/1.1.x/deploying/wsgi-standalone/).

### Set up Nginx

To redirect the traffic to your local app.
Configure your Nginx `.conf` file.

```
server {
  listen  80;

  client_max_body_size 20M;

  location / {
      proxy_pass http://127.0.0.1:5000;
  }
}
```

</details>

## Resources

[Building a simple Keras Deep Learning REST API](https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html)
