pip freeze > requirements.txt
echo "web: python app.py" > Procfile

python app.py

To deploy in "Render"
Build Command ： pip install -r requirements.txt
Start Command ： gunicorn app:app
