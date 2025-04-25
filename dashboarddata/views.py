from django.shortcuts import render
from .models import MyTest
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from django.urls import reverse
from django.shortcuts import redirect

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import numpy as np
import re
from gensim.models.keyedvectors import KeyedVectors


from transformers import BertTokenizer, BertModel
import torch

import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

import tensorflow as tf
import tensorflow_hub as hub
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download the Universal Sentence Encoder model
# module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
# embed = hub.load(module_url)
# Define a function to calculate possible engines

# def calculate_possible_engines(description):
#     mytest_data = MyTest.objects.all()
#     engine_descriptions = {}
#
#     for row in mytest_data:
#         engine = row.L1_Engine
#         description_lower = row.Description.lower()
#         if engine in engine_descriptions:
#             engine_descriptions[engine].append(description_lower)
#         else:
#             engine_descriptions[engine] = [description_lower]
#
#     engine_scores = []
#
#     vectorizer = TfidfVectorizer(stop_words=['the', 'and', 'a', 'marketing', 'agency', 'be', 'an', 'for', 'as', 'to',
#                                              'marketing', 'activities', 'services', 'organization', 'microsoft',
#                                              'deliver', 'prepare', 'company', 'market', 'activity', 'service',
#                                              'organisation', 'ms', 'delivery', 'preparation', 'companies', 'msft',
#                                              'preparing', 'what', 'when', 'where', 'who'])
#
#     # Remove numeric words from the input description and split into words
#     description_words = [word for word in description.split() if not any(c.isdigit() for c in word)]
#
#     for engine, descriptions in engine_descriptions.items():
#         # Calculate context scores and keyword scores for all descriptions for the engine
#         context_scores = []
#         keyword_scores = []
#
#         for desc in descriptions:
#             # Calculate sentence embeddings for the input description and the current description
#             sentence_embeddings = embed([description, desc])
#
#             # Calculate cosine similarity between the sentence embeddings
#             context_score = cosine_similarity([sentence_embeddings[0]], [sentence_embeddings[1]])[0][0]
#             context_scores.append(context_score)
#
#             # Extract keywords from the current description (words with high TF-IDF scores)
#             description_vectors = vectorizer.fit_transform([description, desc])
#             feature_names = vectorizer.get_feature_names_out()
#             keyword_list = [feature_names[i] for i in description_vectors[1].indices]
#
#             # Calculate a keyword-based score based on how many keywords appear in the input description
#             keyword_score = sum(1 for keyword in keyword_list if keyword in description_words)
#             keyword_scores.append(keyword_score)
#
#         # Find the description with the maximum context score
#         max_context_score = max(context_scores)
#
#         # Calculate the keyword score for the description with the maximum context score
#         max_context_index = context_scores.index(max_context_score)
#         max_keyword_score = keyword_scores[max_context_index]
#
#         # Combine context and keyword scores (adjust weights as needed)
#         total_score = (max_context_score * 0.3 + max_keyword_score * 0.7) * 100
#
#         # Ensure that the final confidence score does not exceed 100 percent
#         engine_scores.append((engine, min(total_score, 100.0)))
#
#     # Sort engines by total score in descending order
#     engine_scores.sort(key=lambda x: x[1], reverse=True)
#
#     return engine_scores


def calculate_possible_engines(description):
    mytest_data = MyTest.objects.all()
    engine_descriptions = {}


    for row in mytest_data:
        engine = row.L1_Engine
        description_lower = row.Description.lower()
        if engine in engine_descriptions:
            engine_descriptions[engine].append(description_lower)
        else:
            engine_descriptions[engine] = [description_lower]

    engine_scores = []

    vectorizer = TfidfVectorizer(stop_words=['the', 'and', 'a', 'marketing', 'agency', 'be', 'an', 'for', 'as', 'to',
                                             'marketing', 'activities', 'services', 'organization', 'microsoft',
                                             'deliver', 'prepare', 'company', 'market', 'activity', 'service',
                                             'organisation', 'ms', 'delivery', 'preparation', 'companies', 'msft',
                                             'preparing', 'what', 'when', 'where', 'who'])

    # Remove numeric words from the input description and split into words
    description_words = [word for word in description.split() if not any(c.isdigit() for c in word)]

    for engine, descriptions in engine_descriptions.items():
        # Calculate context scores and keyword scores for all descriptions for the engine
        context_scores = []
        keyword_scores = []

        for desc in descriptions:
            # Calculate TF-IDF vectors for the input description and the current description
            description_vectors = vectorizer.fit_transform([description, desc])

            # Calculate cosine similarity between the input description and the current description
            cosine_similarities = cosine_similarity(description_vectors)

            # Calculate a context-based score as the cosine similarity
            context_score = cosine_similarities[0][1]
            context_scores.append(context_score)

            # Extract keywords from the current description (words with high TF-IDF scores)
            feature_names = vectorizer.get_feature_names_out()
            keyword_list = [feature_names[i] for i in description_vectors[1].indices]

            # Calculate a keyword-based score based on how many keywords appear in the input description
            # Calculate  a keyword-based score based on how many keywords appear in the input description
            # Count the number of keywords that match
            keyword_count = sum(1 for keyword in keyword_list if keyword in description_words)

            # Calculate the maximum possible keyword score (when all keywords match)
            max_keyword_count = len(description_words)

            # Normalize the keyword score to a range between 0 and 1
            if max_keyword_count > 0:
                keyword_score = keyword_count / max_keyword_count
            else:
                keyword_score = 0.0  # Handle the case where there are no keywords in the input description

            keyword_scores.append(keyword_score)

        # Find the description with the maximum context score
        max_context_score = max(context_scores)

        # Calculate the keyword score for the description with the maximum context score
        max_context_index = context_scores.index(max_context_score)
        max_keyword_score = keyword_scores[max_context_index]

        # Combine context and keyword scores (adjust weights as needed)
        total_score = (max_context_score * 0.5 + max_keyword_score * 0.5) * 100

        # Ensure that the final confidence score does not exceed 100 percent
        # total_score = min(total_score, 100.0)

        engine_scores.append((engine, total_score))

    # Sort engines by total score in descending order
    engine_scores.sort(key=lambda x: x[1], reverse=True)

    return engine_scores

# def calculate_possible_engines(description):
#     mytest_data = MyTest.objects.all()
#     engine_descriptions = {}
#
#     # Function to remove words with numeric values
#     def remove_numeric_words(text):
#         return ' '.join(word for word in text.split() if not re.search(r'\d', word))
#
#     for row in mytest_data:
#         engine = row.L1_Engine
#         description_lower = row.Description.lower()
#         if engine in engine_descriptions:
#             engine_descriptions[engine].append(description_lower)
#         else:
#             engine_descriptions[engine] = [description_lower]
#
#     engine_scores = []
#
#     vectorizer = TfidfVectorizer(stop_words=['the', 'and', 'a', 'marketing', 'agency', 'be', 'an', 'for', 'as', 'to',
#                                              'marketing', 'activities', 'services', 'organization', 'microsoft',
#                                              'deliver', 'prepare', 'company', 'market', 'activity', 'service',
#                                              'organisation', 'ms', 'delivery', 'preparation', 'companies', 'msft',
#                                              'preparing'])
#
#     # Remove numeric words from the input description
#     description_words = remove_numeric_words(description.lower())
#
#     for engine, descriptions in engine_descriptions.items():
#         # Remove numeric words from each description
#         descriptions = [remove_numeric_words(d) for d in descriptions]
#
#         description_vectors = vectorizer.fit_transform([description_words] + descriptions)
#         cosine_similarities = cosine_similarity(description_vectors[0], description_vectors[1:]).flatten()
#         max_cosine_sim = max(cosine_similarities)
#
#         engine_scores.append((engine, max_cosine_sim * 100))
#
#     engine_scores.sort(key=lambda x: x[1], reverse=True)
#     return engine_scores


# def calculate_possible_engines(description):
#     mytest_data = MyTest.objects.all()
#     engine_descriptions = {}
#
#     for row in mytest_data:
#         engine = row.L1_Engine
#         description_lower = row.Description.lower()
#         if engine in engine_descriptions:
#             engine_descriptions[engine].append(description_lower)
#         else:
#             engine_descriptions[engine] = [description_lower]
#
#     engine_scores = []
#
#     vectorizer = TfidfVectorizer(stop_words=['the', 'and', 'a','marketing','agency','be','an','for','as','to','marketing ',
# 'activities ',
# 'services ',
# 'organization ',
# 'microsoft ',
# 'deliver ',
# 'prepare ',
# 'company ',
# ' market',
# ' activity',
# ' service',
# ' organisation',
# ' ms ',
# ' delivery',
# ' preparation ',
# ' companies',
# 'msft',
# ' preparing'
# ])
#
#     for engine, descriptions in engine_descriptions.items():
#         description_vectors = vectorizer.fit_transform([description.lower()] + descriptions)
#         cosine_similarities = cosine_similarity(description_vectors[0], description_vectors[1:]).flatten()
#         max_cosine_sim = max(cosine_similarities)
#
#         engine_scores.append((engine, max_cosine_sim * 100))
#
#     engine_scores.sort(key=lambda x: x[1], reverse=True)
#     return engine_scores





def index(request):
    file_uploaded = False
    possible_engines = []
    uploaded_description = request.POST.get('uploaded_description', '')  # Get the uploaded description

    if request.method == 'POST':
        if 'file' in request.FILES:
            file = request.FILES['file']

            delete_all_records()

            # Read and save data from uploaded Excel file
            file1 = pd.read_excel(file, sheet_name="Sheet1")
            file1.columns = [re.sub(r'[\s\xa0]+', '_', col) for col in file1.columns]
            # print(file1.columns)

            for row in file1.itertuples():
                l1_engine = row.L1_Engine
                primary_execution_team = row.Primary_Execution_Team
                description = row.Description

                value = MyTest(
                    L1_Engine=l1_engine,
                    Primary_Execution_Team=primary_execution_team,
                    Description=description
                )
                value.save()

            file_uploaded = True
        elif 'uploaded_description' in request.POST:
            uploaded_description = request.POST.get('uploaded_description', '')
            if uploaded_description:
                possible_engines = calculate_possible_engines(uploaded_description)

    mytest = MyTest.objects.all()

    return render(request, "hello.html",
                  {'var': mytest, 'file_uploaded': file_uploaded, 'possible_engines': possible_engines,'uploaded_description': uploaded_description})

def delete_all_records():
    MyTest.objects.all().delete()
# ... (rest of the code)
