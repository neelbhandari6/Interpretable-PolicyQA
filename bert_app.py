import streamlit as st
from saliency_vis import *
from para_ranker import *
st.set_page_config(layout="wide")
st.markdown("""
<style>
body {
    color: #fff;
    background-color: #4F8BF9;
}
</style>
    """, unsafe_allow_html=True)
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import imgkit
import torch
import torch.nn as nn

from transformers import BertTokenizer, BertForQuestionAnswering, BertConfig,DistilBertForQuestionAnswering

from captum.attr import visualization as viz
from captum.attr import IntegratedGradients, LayerConductance, LayerIntegratedGradients, LayerActivation
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



st.title("Interpretable Question Answering For Privacy Policy")
def load_model():
    # model_path = '/Users/neelbhandari/Downloads/distilbert_weights'
    model_path='distilbert_weights'
# load model
    model = DistilBertForQuestionAnswering.from_pretrained(model_path)
    model.to(device)
    model.eval()
    model.zero_grad()
    return model

def load_tokeniser():
    tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')
    return tokenizer

# model=load_model()
# tokenizer=load_tokeniser()

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert_qa")


def answer_question(model,question:str,text:str):
    input_ids, ref_input_ids, sep_id = construct_input_ref_pair(question, text, ref_token_id, sep_token_id, cls_token_id)
    token_type_ids, ref_token_type_ids = construct_input_ref_token_type_pair(input_ids, sep_id)
    position_ids, ref_position_ids = construct_input_ref_pos_id_pair(input_ids)
    attention_mask = construct_attention_mask(input_ids)

    indices = input_ids[0].detach().tolist()
    all_tokens = tokenizer.convert_ids_to_tokens(indices)
    indices = input_ids[0].detach().tolist()
    results= predict(input_ids, \
                                   attention_mask=attention_mask)
    output_attentions=results['attentions']
                                   
    x=' '.join(all_tokens[torch.argmax(results['start_logits'][0]) : torch.argmax(results['end_logits'][0])+1])
    return x,all_tokens,indices,input_ids,ref_input_ids,attention_mask,results,output_attentions

def predict(inputs, position_ids=None, attention_mask=None):
    return model(inputs,
                 attention_mask=attention_mask,output_attentions=True )


def squad_pos_forward_func(inputs, attention_mask=None, position=0):
    pred = predict(inputs,
                   attention_mask=attention_mask)
    pred = pred[position]
    return pred.max(1).values

ref_token_id = tokenizer.pad_token_id # A token used for generating token reference
sep_token_id = tokenizer.sep_token_id # A token used as a separator between question and text and it is also added to the end of the text.
cls_token_id = tokenizer.cls_token_id # A token used for prepending to the concatenated question-text word sequence
def construct_input_ref_pair(question, text, ref_token_id, sep_token_id, cls_token_id):
    question_ids = tokenizer.encode(question, add_special_tokens=False)
    text_ids = tokenizer.encode(text, add_special_tokens=False)

    # construct input token ids
    input_ids = [cls_token_id] + question_ids + [sep_token_id] + text_ids + [sep_token_id]

    # construct reference token ids 
    ref_input_ids = [cls_token_id] + [ref_token_id] * len(question_ids) + [sep_token_id] + \
        [ref_token_id] * len(text_ids) + [sep_token_id]

    return torch.tensor([input_ids], device=device), torch.tensor([ref_input_ids], device=device), len(question_ids)

def construct_input_ref_token_type_pair(input_ids, sep_ind=0):
    seq_len = input_ids.size(1)
    token_type_ids = torch.tensor([[0 if i <= sep_ind else 1 for i in range(seq_len)]], device=device)
    ref_token_type_ids = torch.zeros_like(token_type_ids, device=device)# * -1
    return token_type_ids, ref_token_type_ids

def construct_input_ref_pos_id_pair(input_ids):
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    # we could potentially also use random permutation with `torch.randperm(seq_length, device=device)`
    ref_position_ids = torch.zeros(seq_length, dtype=torch.long, device=device)

    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
    return position_ids, ref_position_ids
    
def construct_attention_mask(input_ids):
    return torch.ones_like(input_ids)

# def construct_bert_sub_embedding(input_ids, ref_input_ids,
#                                    token_type_ids, ref_token_type_ids,
#                                    position_ids, ref_position_ids):
#     input_embeddings = interpretable_embedding1.indices_to_embeddings(input_ids)
#     ref_input_embeddings = interpretable_embedding1.indices_to_embeddings(ref_input_ids)

#     input_embeddings_token_type = interpretable_embedding2.indices_to_embeddings(token_type_ids)
#     ref_input_embeddings_token_type = interpretable_embedding2.indices_to_embeddings(ref_token_type_ids)

#     input_embeddings_position_ids = interpretable_embedding3.indices_to_embeddings(position_ids)
#     ref_input_embeddings_position_ids = interpretable_embedding3.indices_to_embeddings(ref_position_ids)
    
#     return (input_embeddings, ref_input_embeddings), \
#            (input_embeddings_token_type, ref_input_embeddings_token_type), \
#            (input_embeddings_position_ids, ref_input_embeddings_position_ids)
    
# def construct_whole_bert_embeddings(input_ids, ref_input_ids, \
#                                     token_type_ids=None, ref_token_type_ids=None, \
#                                     position_ids=None, ref_position_ids=None):
#     input_embeddings = interpretable_embedding.indices_to_embeddings(input_ids)
#     ref_input_embeddings = interpretable_embedding.indices_to_embeddings(ref_input_ids)
    
#     return input_embeddings, ref_input_embeddings



def visualize_token2token_scores(scores_mat, x_label_name='Head'):
    fig = plt.figure(figsize=(20, 20))

    for idx, scores in enumerate(scores_mat):
        scores_np = np.array(scores)
        ax = fig.add_subplot(4, 3, idx+1)
        # append the attention weights
        im = ax.imshow(scores, cmap='viridis')

        fontdict = {'fontsize': 10}

        ax.set_xticks(range(len(all_tokens)))
        ax.set_yticks(range(len(all_tokens)))

        ax.set_xticklabels(all_tokens, fontdict=fontdict, rotation=90)
        ax.set_yticklabels(all_tokens, fontdict=fontdict)
        ax.set_xlabel('{} {}'.format(x_label_name, idx+1))

        fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()
    st.pyplot(fig)

def visualize_token2head_scores(scores_mat):
    fig = plt.figure(figsize=(30, 50))

    for idx, scores in enumerate(scores_mat):
        scores_np = np.array(scores)
        ax = fig.add_subplot(6, 2, idx+1)
        # append the attention weights
        im = ax.matshow(scores_np, cmap='viridis')

        fontdict = {'fontsize': 20}

        ax.set_xticks(range(len(all_tokens)))
        ax.set_yticks(range(len(scores)))

        ax.set_xticklabels(all_tokens, fontdict=fontdict, rotation=90)
        ax.set_yticklabels(range(len(scores[0])), fontdict=fontdict)
        ax.set_xlabel('Layer {}'.format(idx+1))

        fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

# layer = 11
def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions

def plot_gradients(tokens, token_types, gradients, title): 
  """ Plot  explanations
  """
  fig= plt.figure(figsize=(21,3)) 
#   ax = fig.add_subplot(1,1,1)
  xvals = [ x + str(i) for i,x in enumerate(tokens)]
  colors =  [ (0,0,1, c) for c,t in zip(gradients, token_types) ]
  edgecolors = [ "black" if t==0 else (0,0,1, c)  for c,t in zip(gradients, token_types) ]
  # colors =  [  ("r" if t==0 else "b")  for c,t in zip(gradients, token_types) ]
  plt.tick_params(axis='both', which='minor', labelsize=29)
  plt.bar(xvals, gradients, color=colors, linewidth=1, edgecolor=edgecolors)
  plt.title(title) 
  plt.xticks(ticks=[i for i in range(len(tokens))], labels=tokens, fontsize=12,rotation=90) 
  st.write(fig)    
def get_global_scores(paras,question):
    global scores
    scores=get_ranks(paras,question)

def get_paras(context):
    global paras
    iparas=context.split("\n")
    paras=[]
    for ipara in iparas:
        if(len(ipara)<512):
            paras.append(ipara)
        else:

            
            for a in range(0,len(ipara),512):
                paras.append(ipara[a:a+512])


def get_preds(question,paras,scores):
    global max_i,max_pred,max_start
    for i in range(10):
        prediction,all_tokens,indices,input_ids,ref_input_ids,attention_mask,results,output_attentions=answer_question(model,question,paras[scores[i]])
        curr=torch.max(results['start_logits'][0])
        if(curr>max_start):
            max_pred=prediction
            max_start=curr
            max_i=i
    if(max_pred==""):
        max_pred=prediction
if __name__ == "__main__":
    
    form = st.form(key='my-form')
    context = form.text_input('Enter the context')
    question=form.text_input('Enter the question')
    submit = form.form_submit_button('Submit')

    max_i=0
    max_start=0
    max_pred=""
    paras=[]
    scores=[]
        

    
    # context= st.text_input("Enter an Excerpt of the Privacy Policy Document Below", "")
    # question = st.text_input("Enter Your Question Below", "")
    if submit:
        

        get_paras(context)
        get_global_scores(paras,question)
        
    
        st.write('Response:')
        # paragraph_slot1 = st.empty()
        # paragraph_slot2 = st.empty()
        # paragraph_slot3 = st.empty()
        # paragraph_slot4 = st.empty()
        # paragraph_slot5 = st.empty()
        with st.spinner('Searching For Your Answer..'):
            
            get_preds(question,paras,scores)
            st.write(max_pred)

            

    if(st.button('See Gradient based attribution')):
        get_paras(context)
        get_global_scores(paras,question)
        max_i=0
        max_start=0
        max_pred=""
        get_preds(question,paras,scores)
        gradients, tokens, token_types, answer  = explain_model(question, paras[scores[max_i]])
        gradients, tokens, token_types = clean_tokens(gradients, tokens, token_types)
        
        qt=question.split()
        q=len(qt)
        st.write("For question:")
        plot_gradients(tokens[:q], token_types[:q], gradients[:q], "Q: " +question + " | A: "+ answer)
        st.write("For context:")
        for i in range(len(qt),len(tokens),60):
            plot_gradients(tokens[i:i+60], token_types[i:i+60], gradients[i:i+60], "Q: " +question + " | A: "+ answer)

                    
            # ground_truth = 'uses your IP address to suggest relevant content based on your country and state'
            # ground_truth_tokens = tokenizer.encode(ground_truth, add_special_tokens=True)
            # ground_truth_end_ind = indices.index(ground_truth_tokens[-1])
            # ground_truth_start_ind = ground_truth_end_ind - len(ground_truth_tokens) + 1

#             lig = LayerIntegratedGradients(squad_pos_forward_func, model.distilbert.embeddings)

#             attributions_start, delta_start = lig.attribute(inputs=input_ids,
#                                   baselines=ref_input_ids,
#                                   additional_forward_args=(attention_mask, 0),
#                                   return_convergence_delta=True)
#             attributions_end, delta_end = lig.attribute(inputs=input_ids, baselines=ref_input_ids,
#                                 additional_forward_args=(attention_mask, 1),
#                                 return_convergence_delta=True)
#             attributions_start_sum = summarize_attributions(attributions_start)
#             attributions_end_sum = summarize_attributions(attributions_end)
#             start_position_vis = viz.VisualizationDataRecord(
#                         attributions_start_sum,
#                         torch.max(torch.softmax(results['start_logits'][0], dim=0)),
#                         torch.argmax(results['start_logits']),
#                         torch.argmax(results['start_logits']),
#                         # str(ground_truth_start_ind),
#                         attributions_start_sum.sum(),       
#                         all_tokens,
#                         delta_start)
#             print(start_position_vis)
#             end_position_vis = viz.VisualizationDataRecord(
#                         attributions_end_sum,
#                         torch.max(torch.softmax(results['end_logits'][0], dim=0)),
#                         torch.argmax(results['end_logits']),
#                         torch.argmax(results['end_logits']),
#                         # str(ground_truth_end_ind),
#                         attributions_end_sum.sum(),       
#                         all_tokens,
#                         delta_end)
#             st.write('\033', 'Visualizations For Start Position', '\033')
#             x=viz.visualize_text([start_position_vis])
#             import streamlit.components.v1 as components  # Import Streamlit

# # Render the h1 block, contained in a frame of size 200x200.
#             components.html(x.data,height=300)
#             st.write('\033[1m', 'Visualizations For End Position', '\033[0m')
#             y=viz.visualize_text([end_position_vis])
#             components.html(y.data,height=300)

#             output_attentions_all=torch.stack(output_attentions)
#             visualize_token2token_scores(output_attentions_all[5].squeeze().detach().cpu().numpy())
