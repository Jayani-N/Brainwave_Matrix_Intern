 #app.py

    from flask import Flask, request, jsonify, render_template
    from flask_cors import CORS
    import re
    
    def check_password_strength(password):
        strength = 0
        feedback = []
    
        if len(password) >= 12:
            strength += 2
        elif len(password) >= 8:
            strength += 1
        else:
            feedback.append("Password should be at least 8 characters long.")
    
        if re.search(r'[A-Z]', password):
            strength += 1
        else:
            feedback.append("Password should contain at least one uppercase letter.")
        if re.search(r'[a-z]', password):
            strength += 1
        else:
            feedback.append("Password should contain at least one lowercase letter.")
        if re.search(r'[0-9]', password):
            strength += 1
        else:
            feedback.append("Password should contain at least one digit.")
        if re.search(r'[@$!%*?&#]', password):
            strength += 1
        else:
            feedback.append("Password should contain at least one special character.")
    
        return {"strength": strength, "feedback": feedback}
    
    app = Flask(__name__)
    CORS(app)
    
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/check_password', methods=['POST'])
    def check_password():
        data = request.get_json()
        password = data.get('password', '')
        result = check_password_strength(password)
        return jsonify(result)
    
    if __name__ == '__main__':
        app.run(debug=True)


# templates\index.html

    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Password Strength Checker</title>
        <link rel="stylesheet" href="{{ url_for('static', filename='styles.css') }}">
    </head>
    <body>
        <div class="container">
            <h1>Password Strength Checker</h1>
            <form id="password-form">
                <input type="password" id="password" placeholder="Enter your password">
                <button type="submit">Check Strength</button>
            </form>
            <div id="result"></div>
        </div>
        <script src="{{ url_for('static', filename='script.js') }}"></script>
    </body>
    </html>



# static\styles.css

    body {
        font-family: Arial, sans-serif;
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
        background-color: #f4f4f4;
    }
    
    .container {
        background: white;
        padding: 2rem;
        border-radius: 8px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    input {
        width: 80%;
        padding: 0.5rem;
        margin-bottom: 1rem;
    }
    
    button {
        padding: 0.5rem 1rem;
        background: #007bff;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }
    
    button:hover {
        background: #0056b3;
    }
    
    #result {
        margin-top: 1rem;
    }



# static\script.js

    document.getElementById('password-form').addEventListener('submit', function(e) {
        e.preventDefault();
        const password = document.getElementById('password').value;
    
        fetch('/check_password', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ password })
        })
        .then(response => response.json())
        .then(data => {
            let resultDiv = document.getElementById('result');
            resultDiv.innerHTML = `
                <p>Strength: ${data.strength}</p>
                ${data.feedback.map(item => `<p>${item}</p>`).join('')}
            `;
        });
    });

# RUNNING THE app.py FILE, THE OUTPUT:

![Screenshot 1](https://github.com/user-attachments/assets/9a85ae66-9026-4f37-9e0d-ab717f502066)

![Screenshot 2](https://github.com/user-attachments/assets/9b40d004-eb03-44b9-a317-5d07ab7f4d90)

![Screenshot 3](https://github.com/user-attachments/assets/f01cf4e0-0ce1-43cb-84af-02caa3d49ebf)

![Screenshot 4](https://github.com/user-attachments/assets/139044d7-0821-4823-8a03-b82982770747)





