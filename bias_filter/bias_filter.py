import torch
import numpy as np
import pandas as pd
import csv
import gdown
import re
import spacy
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from torch.utils.data import DataLoader
from transformers import (
    BertTokenizer, BertModel, BertForMaskedLM, BertTokenizerFast, pipeline,
    AutoTokenizer, AutoModelForSeq2SeqLM, AdamW,
    get_linear_schedule_with_warmup
)
from Dbias.text_debiasing import *
from Dbias.bias_classification import *
from Dbias.bias_recognition import *
from Dbias.bias_masking import *
from Dbias.bias_classification import classifier

# Download the 'wordnet' dataset
nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")

class BiasFilter:
  PRONOUNS = {
        "he": "they", "she": "they",
        "him": "them", "her": "them",
        "his": "their", "hers": "theirs",
        "himself": "themselves", "herself": "themselves",
        "s/he": "they", "xe": "they", "ze": "they", "ey": "they"
    }

  GENDERED_NOUNS = {
        "husband": "spouse", "wife": "spouse",
        "father": "parent", "mother": "parent",
        "son": "child", "daughter": "child",
        "brother": "sibling", "sister": "sibling",
        "uncle": "relative", "aunt": "relative",
        "nephew": "relative", "niece": "relative",
        "grandfather": "grandparent", "grandmother": "grandparent",
        "fiancé": "partner", "fiancée": "partner",
        "boy": "child", "girl": "child",
        "stepfather": "stepparent", "stepmother": "stepparent",
        "stepson": "stepchild", "stepdaughter": "stepchild",
        "godfather": "godparent", "godmother": "godparent",
        "boyfriend": "partner", "girlfriend": "partner",
        "bride": "married partner", "groom": "married partner",
        "fireman": "firefighter", "policeman": "police officer",
        "mailman": "mail carrier", "salesman": "salesperson",
        "businessman": "businessperson", "chairman": "chairperson",
        "waiter": "server", "waitress": "server",
        "steward": "flight attendant", "stewardess": "flight attendant",
        "actor": "performer", "actress": "performer",
        "congressman": "legislator", "congresswoman": "legislator",
        "alderman": "council member", "ombudsman": "ombudsperson",
        "postman": "postal worker", "weatherman": "meteorologist",
        "craftsman": "artisan", "foreman": "supervisor",
        "milkman": "milk deliverer", "fisherman": "fisher",
        "draftsman": "drafter", "seamstress": "sewer",
        "laundryman": "laundry worker", "repairman": "repair technician",
        "watchman": "security guard", "middleman": "intermediary",
        "workman": "worker", "spokesman": "spokesperson",
        "forefather": "ancestor", "maid": "housekeeper",
        "housewife": "homemaker", "househusband": "homemaker",
        "bondsman": "guarantor", "clergyman": "clergy",
        "policewoman": "police officer", "handyman": "maintenance worker",
        "showman": "entertainer", "cameraman": "camera operator",
        "prince": "royal", "princess": "royal",
        "king": "monarch", "queen": "monarch",
        "duke": "noble", "duchess": "noble",
        "emperor": "ruler", "empress": "ruler",
        "lord": "noble", "lady": "noble",
        "sultan": "ruler", "sultana": "ruler",
        "serviceman": "service member", "servicemen": "service members",
        "servicelady": "service member", "serviceladies": "service members",
        "airman": "aviator", "seaman": "sailor",
        "infantryman": "infantry soldier", "guardsman": "guard",
        "rifleman": "sharpshooter", "midshipman": "naval officer",
        "monk": "clergy", "nun": "clergy",
        "priest": "clergy", "priestess": "clergy",
        "chaplain": "spiritual leader",
        "sportsman": "athlete", "sportswoman": "athlete",
        "batman": "cricket assistant", "batboy": "equipment manager",
        "linesman": "referee", "ballboy": "ball retriever",
        "headmaster": "principal", "headmistress": "principal",
        "councilman": "council member", "councilwoman": "council member",
        "founding fathers": "founders", "manpower": "workforce",
        "mankind": "humanity", "brotherhood": "fellowship",
        "fellow": "peer", "grandmaster": "expert",
        "layman": "non-specialist", "marksman": "sharpshooter",
        "newsman": "journalist", "nobleman": "noble",
        "playboy": "socialite", "showgirl": "performer",
        "strongman": "weightlifter", "tradesman": "trader",
        "workman": "worker", "yachtsman": "sailor",
        "drummer boy": "drummer", "peasant woman": "farmer",
        "gentleman": "person", "lady": "person",
        "grandson": "grandchild", "granddaughter": "grandchild",
        "bachelor": "unmarried person", "spinster": "unmarried person",
        "manhunt": "search", "policemen": "police officers",
        "bridegroom": "married partner", "horseman": "rider",
        "journeyman": "skilled worker", "lumberjack": "logger",
        "midwife": "birth assistant", "tomboy": "energetic child",
        "wise man": "sage", "witch": "magic practitioner",
        "wizard": "magic practitioner", "heir": "successor",
        "heiress": "successor"
    }
  def __init__(self):
    # Load the pre-trained BERT model and tokenizer
    self.tokenizer = BertTokenizerFast.from_pretrained("kdadobe1/bert-bias-detection-retrained")
    self.bert_model = BertForMaskedLM.from_pretrained("kdadobe1/bert-bias-detection-retrained")
    self.t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")  # Or a larger t5_model
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.t5_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    self.t5_model.to(self.device) 

  def complete_statement(self, masked_statement):
    """Predicts the masked word(s) in the input statement."""
    inputs = self.tokenizer(masked_statement, return_tensors="pt")
    with torch.no_grad():
        outputs = self.bert_model(**inputs)
    predictions = outputs.logits
    # Get the predicted token index for the [MASK] token
    mask_token_index = torch.where(inputs.input_ids == self.tokenizer.mask_token_id)[1]
    predicted_token_id = predictions[0, mask_token_index, :].argmax(axis=-1)
    # Replace the [MASK] token with the predicted word
    predicted_token = self.tokenizer.decode(predicted_token_id)
    completed_statement = masked_statement.replace(self.tokenizer.mask_token, predicted_token)
    return completed_statement, predicted_token
  def is_biased(self, statement):

    """Uses the dbias classifier to check if the statement is biased."""
    biased_result = classifier(statement)
    print(biased_result)
    return biased_result[0]['label'] == 'Biased'

  def rephrase_with_t5(self, sentence):
    """Rephrases the sentence in detoxified, gender-neutral, and proper English format."""

    # Stronger and clearer prompt
    input_text = f"Paraphrase this sentence in fluent, detoxified English: {sentence}"

    input_ids = self.t5_tokenizer(input_text, return_tensors="pt").input_ids
    input_ids = input_ids.to(self.device) 
    
    outputs = self.t5_model.generate(
        input_ids,
        max_length=100,
        min_length=10,
        temperature=0.7,
        top_p=0.9,
        num_beams=5
    )

    # Decode and clean up the output
    rephrased_text = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print(rephrased_text)

    # Remove any unwanted repetition of the input prompt
    if rephrased_text.lower().startswith("paraphrase this"):
        rephrased_text = rephrased_text.split(":", 1)[-1].strip()

    return rephrased_text


  def rephrase_with_combined(self,sentence):
      """ Makes sentence gender-neutral using regex, dictionary replacement, and T5 (optional). """
      doc = nlp(sentence[:1000])
      new_sentence = []
      PRONOUN_SET = set(BiasFilter.PRONOUNS.keys())
      GENDERED_NOUN_SET = set(BiasFilter.GENDERED_NOUNS.keys())
      for token in doc:
          word_lower = token.text.lower()
          if word_lower in BiasFilter.PRONOUNS:
              new_sentence.append(BiasFilter.PRONOUNS[word_lower])
          elif token.pos_ == "NOUN" and word_lower in BiasFilter.GENDERED_NOUNS:
              new_sentence.append(BiasFilter.GENDERED_NOUNS[word_lower])
          else:
              new_sentence.append(token.text)
      neutral_sentence = " ".join(new_sentence)
      neutral_sentence = self.rephrase_with_t5(neutral_sentence)
      return neutral_sentence

  def apply_debiasing(self,sentence):
    output = run(sentence, show_plot = False)
    if output and len(output) > 0:
      sentence = output[0]['Sentence']
    return sentence
  def process_statement(self, masked_statement):
      """Handles the full workflow: completion, bias detection, and correction."""
      completed_statement, predicted_token = self.complete_statement(masked_statement)
      print(f"Completed Statement: {completed_statement}")
      print(f"Predicted token: {predicted_token}")
      biased = self.is_biased(completed_statement)
      if biased:
          print("******The statement is biased*******")
          print("******Applying debiasing*******")
          debiased_statement = self.apply_debiasing(completed_statement)
          print("******Applying gender-neutral filter.******")
          gender_neutral_statement = self.rephrase_with_combined(debiased_statement)
          return gender_neutral_statement
      else:
          print("The statement is not biased.")
          return completed_statement
