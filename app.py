from flask import Flask, render_template_string, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("decision_model.pkl")

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>Digital Wellness Predictor</title>

    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
        }

        body {
            min-height: 100vh;
            background: linear-gradient(135deg, #667eea, #764ba2);
            padding: 30px 15px;
        }

        .container {
            max-width: 900px;
            margin: auto;
        }

        .header {
            text-align: center;
            color: white;
            margin-bottom: 25px;
        }

        .header h1 {
            font-size: 38px;
            margin-bottom: 10px;
        }

        .header p {
            font-size: 17px;
            opacity: 0.9;
        }

        .card {
            background: white;
            border-radius: 20px;
            padding: 35px;
            box-shadow: 0 15px 40px rgba(0,0,0,0.20);
        }

        .section-title {
            font-size: 22px;
            color: #333;
            margin-bottom: 20px;
            border-left: 5px solid #667eea;
            padding-left: 12px;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        label {
            font-weight: bold;
            color: #444;
            margin-bottom: 8px;
        }

        input, select {
            padding: 13px;
            border: 1px solid #ddd;
            border-radius: 10px;
            font-size: 15px;
            outline: none;
            transition: 0.3s;
        }

        input:focus, select:focus {
            border-color: #667eea;
            box-shadow: 0 0 5px rgba(102,126,234,0.3);
        }

        .button-area {
            text-align: center;
            margin-top: 30px;
        }

        button {
            width: 100%;
            padding: 15px;
            border: none;
            border-radius: 12px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            transition: 0.3s;
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(102,126,234,0.4);
        }

        .result {
            margin-top: 30px;
            padding: 22px;
            text-align: center;
            border-radius: 15px;
            background: #f3f4ff;
        }

        .result h2 {
            color: #667eea;
            margin-bottom: 8px;
        }

        .result p {
            font-size: 18px;
            color: #333;
        }

        .footer {
            text-align: center;
            color: white;
            margin-top: 20px;
            font-size: 14px;
        }

        @media(max-width: 700px) {
            .form-grid {
                grid-template-columns: 1fr;
            }

            .header h1 {
                font-size: 30px;
            }

            .card {
                padding: 25px;
            }
        }
    </style>
</head>

<body>

<div class="container">

    <div class="header">
        <h1>📱 Digital Wellness Predictor</h1>
        <p>Decision Tree based prediction system</p>
    </div>

    <div class="card">

        <form method="POST">

            <div class="section-title">
                👤 Personal Information
            </div>

            <div class="form-grid">

                <div class="form-group">
                    <label>Age</label>
                    <input type="number" name="age" min="1" required>
                </div>

                <div class="form-group">
                    <label>Gender</label>
                    <select name="gender" required>
                        <option value="">Select Gender</option>
                        <option value="0">Female</option>
                        <option value="1">Male</option>
                    </select>
                </div>

            </div>

            <br>

            <div class="section-title">
                📱 Screen & App Usage
            </div>

            <div class="form-grid">

                <div class="form-group">
                    <label>Daily Screen Time (Hours)</label>
                    <input type="number" name="daily_screen_time_hours"
                           step="0.1" min="0" required>
                </div>

                <div class="form-group">
                    <label>Social Media Hours</label>
                    <input type="number" name="social_media_hours"
                           step="0.1" min="0" required>
                </div>

                <div class="form-group">
                    <label>Gaming Hours</label>
                    <input type="number" name="gaming_hours"
                           step="0.1" min="0" required>
                </div>

                <div class="form-group">
                    <label>Work / Study Hours</label>
                    <input type="number" name="work_study_hours"
                           step="0.1" min="0" required>
                </div>

                <div class="form-group">
                    <label>Sleep Hours</label>
                    <input type="number" name="sleep_hours"
                           step="0.1" min="0" required>
                </div>

                <div class="form-group">
                    <label>Notifications Per Day</label>
                    <input type="number" name="notifications_per_day"
                           min="0" required>
                </div>

                <div class="form-group">
                    <label>App Opens Per Day</label>
                    <input type="number" name="app_opens_per_day"
                           min="0" required>
                </div>

                <div class="form-group">
                    <label>Weekend Screen Time</label>
                    <input type="number" name="weekend_screen_time"
                           step="0.1" min="0" required>
                </div>

            </div>

            <br>

            <div class="section-title">
                🧠 Mental & Academic Factors
            </div>

            <div class="form-grid">

                <div class="form-group">
                    <label>Stress Level</label>
                    <input type="number" name="stress_level"
                           min="0" max="10" required>
                </div>

                <div class="form-group">
                    <label>Academic Work Impact</label>
                    <input type="number" name="academic_work_impact"
                           min="0" max="10" required>
                </div>

                <div class="form-group">
                    <label>Addiction Level</label>
                    <input type="number" name="addiction_level"
                           min="0" max="10" required>
                </div>

            </div>

            <div class="button-area">
                <button type="submit">
                    🔮 Predict Result
                </button>
            </div>

        </form>

        {% if prediction is not none %}
        <div class="result">
            <h2>Prediction Result</h2>
            <p><strong>Predicted Class:</strong> {{ prediction }}</p>
        </div>
        {% endif %}

    </div>

    <div class="footer">
        © 2026 Digital Wellness Prediction | Powered by Flask & Machine Learning
    </div>

</div>

</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        try:
            data = [
                float(request.form["age"]),
                float(request.form["gender"]),
                float(request.form["daily_screen_time_hours"]),
                float(request.form["social_media_hours"]),
                float(request.form["gaming_hours"]),
                float(request.form["work_study_hours"]),
                float(request.form["sleep_hours"]),
                float(request.form["notifications_per_day"]),
                float(request.form["app_opens_per_day"]),
                float(request.form["weekend_screen_time"]),
                float(request.form["stress_level"]),
                float(request.form["academic_work_impact"]),
                float(request.form["addiction_level"])
            ]

            prediction = model.predict([data])[0]

        except Exception as e:
            prediction = "Error: " + str(e)

    return render_template_string(
        HTML,
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
