"""
Script to create a web application that wraps the trained model to be used for inference using Flask.
The front-end is designed in `./templates/page.html` and its styles in `./static/page.css`
"""

from flask import Flask, abort, flash, render_template, request
from omegaconf import OmegaConf

from src import data
from src.exception_handler import NotFoundError
from src.inference import KeywordSpotter

app = Flask(__name__)
cfg = OmegaConf.load("./config_dir/config.yaml")


@app.route("/")
def home():
    """
    Returns the result of calling render_template() with page.html
    """
    return render_template("page.html")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """
    Returns the prediction from trained model artifact whenever transcribe route is called.
    It accepts file input (.wav) whenever user uploads the file, and make prediction using it.
    The `app.route()` decorator does the job of event handling by means of `jinja2` template
    engine.

    Raises
    ------
    NotFoundError: Exception
        404 error, if any exception occurs.
    """

    recognized_keyword = ""
    if request.method == "POST":
        audio_file = request.files["file"]
        if audio_file.filename == "":
            flash("File not found !!!", category="error")
            return render_template("page.html")

        elif not data.check_fileType(filename=audio_file.filename, extension=".wav"):
            flash(
                "Unsupported file format. Please use only .wav files", category="error"
            )
            return render_template("page.html")

        else:
            try:
                recognizer = KeywordSpotter(
                    audio_file,
                    cfg.paths.model_artifactory_dir,
                    cfg.params.n_mfcc,
                    cfg.params.mfcc_length,
                    cfg.params.sampling_rate,
                )
                recognized_keyword, label_probability = recognizer.predict()

            except NotFoundError:
                abort(
                    404,
                    description="Sorry, something went wrong. Cannot predict from the model. Please try again !!!",
                )

    return render_template(
        "page.html",
        recognized_keyword=f"Transcribed keyword: {recognized_keyword.title()}",
        label_probability=f"Predicted probability: {label_probability}",
    )


if __name__ == "__main__":
    app.run(debug=False)
