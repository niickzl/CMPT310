# CMPT310
CMPT 310 project

FOR TESTERS:
Open Chrome -> go to ```chrome://extensions/``` -> toggle developer mode "ON"
-> Click on "Load unpacked" -> Select the local `extension` folder.

FOR MODERATORS:
Run ```python -m venv .venv``` in terminal.

(RUN THIS EVERYTIME YOU WORK ON THIS)
Then run ```.venv/Scripts/activate``` (Windows)
         ```source .venv/bin/active``` (Mac/Linux)

Then run ```pip install fastapi uvicorn spacy scikit-learn torch transformers```

Then run ```python -m spacy download en_core_web_sm```