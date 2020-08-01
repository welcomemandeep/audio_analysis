import logging

from RenderEngine.QuestionTypeDetection import QuestionTypeDetection
from RenderEngine.LanguageParser import GetNER
from RenderEngine.IsQuestion import IsQuestion

from flask import Flask, render_template, request
from flask_restful import Resource, Api
from flask import jsonify, make_response

logger = logging.getLogger(__name__)

default_participants = ["Matt", "Mike", "Sal", "Rajesh", "Dora", "Sarah"]
participants = default_participants


def server():
    detect_question = QuestionTypeDetection("SVM")
    get_ner = GetNER()
    is_q = IsQuestion()
    app = Flask(__name__)
    api = Api(app)

    @app.route("/")
    def home():
        global participants
        return render_template("index.html", participants=participants)

    @app.route("/participants", methods=["POST", "GET"])
    def participants():
        global participants
        participants_raw = request.form["participants"]
        if (len(participants_raw) > 0):
            participants_raw = str.replace(str.strip(participants_raw), "\n", "")
            participants_raw = participants_raw.split(",")
            participants_raw = [str.strip(participant) for participant in participants_raw]
        participants = participants_raw
        data = {'message': 'Created', 'code': 'SUCCESS'}
        return make_response(jsonify(data), 201)

    @app.route("/reset", methods=["GET"])
    def reset():
        global participants
        participants = default_participants
        return render_template("index.html", participants=participants)

    @app.route('/predict', methods=["POST"])
    def predict():
        global participants
        sentence = request.form['text']
        nouns = get_ner.get_nouns(sentence)
        nouns = [x.lower() for x in nouns]
        lower_case_participants = [x.lower() for x in participants]
        question_type = detect_question.predict(sentence)
        is_question = is_q.predict_question(sentence)
        found_participants = set(lower_case_participants).intersection(nouns)
        if len(found_participants) == 0:
            found_participants = ["everyone/open"]
        found_participants = [x.capitalize() for x in found_participants]
        return render_template("index.html", participants=participants, is_question=is_question,
                               question_type=question_type, subject=found_participants, context=sentence)

    # Run server
    app.run(host='0.0.0.0')
    logger.info("Server started")
