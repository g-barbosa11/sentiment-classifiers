# Description
This project pertains to my undergraduate thesis conducted at the Federal University of São Carlos, focusing on sentiment analysis. The code contained in this repository relates to the training of three language models: BERT, RoBERTa and BERTimbau.


# Project Setup
Runs this  commands to setting up the enverioment:

- `python3 -m venv venv && source venv/bin/activate`
- `pip install -r requirements.txt`

# Running the model
To run the code, simply insert the training, testing, and validation datasets, and then update the model_name variable in the main.py file. After making these changes, execute the main.py file by running the command: 

```bash
python3 model/main.py
``````