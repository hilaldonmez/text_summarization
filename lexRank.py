import re
import os
from collections import defaultdict
import math
import numpy as np
from collections import Counter
from numpy import linalg as LA
from rouge import Rouge
import sys

stopwords_path="stopwords.txt" 

threshold=0.1
eps=0.00001
telep_rate=0.15
#%%
def fill_stopword(stopwords_path):    
    stoplist=[]
    file = open(stopwords_path, 'r') 
    for line in file: 
        token=re.split('\n',line)
        stoplist.append(token[0].strip()) 
    file.close()
    return stoplist

#%%
def read_all_file(full_file_path,postings,doc_sentences,stoplist):
    count=0
    filename_docID={}
    for filename in sorted(os.listdir(full_file_path)):
        filename_docID[filename]=count
        read_file(count,full_file_path+"/"+filename,stoplist,postings,doc_sentences)
        count+=1
    N=count    
    return filename_docID,N    
#%%
def preprocessing(docID,all_sentences,stoplist,postings):
    sentences=[]
    gold_summary_index=all_sentences.index('\n')
    doc_stem=defaultdict() 
    
    for i,each_sentence in enumerate(all_sentences[0:gold_summary_index]): 
        each_sentence = re.sub('\n', '', each_sentence)
        vocabulary=re.findall(r'(?ms)\W*(\w+)', each_sentence)
        vocabulary=[s.lower() for s in vocabulary if s.lower() not in stoplist and len(s)>1 and s.isalpha()]
        
        
        sentences.append(vocabulary)
        for each_stem in vocabulary:
            doc_stem[each_stem]=1
        
    for each_stem in doc_stem:
        if each_stem in postings:
            postings[each_stem].append(docID)
        else:
            postings[each_stem]=[docID]

    return sentences,postings
    
#%%
def read_file(docID,file_path,stoplist,postings,doc_sentences):
    with open(file_path,encoding='iso-8859-1') as file:  
        all_sentences = file.readlines()       
        sentences,postings=preprocessing(docID,all_sentences,stoplist,postings)
        doc_sentences[docID]=sentences
        
#%%
def final_dictionary(distinct_stem):
    dictionary_list=list(distinct_stem.keys())
    for each_stem in distinct_stem:
        distinct_stem[each_stem]=dictionary_list.index(each_stem)

    return distinct_stem
        
#%%
        
def calculate_tfidf(N,postings,doc_sentences):
    
    temp=[]
    for each_sentence in doc_sentences:
        temp_sentence={}
        sentence_counter=Counter(each_sentence)
        for each_stem in sentence_counter:
            temp_sentence[each_stem]=sentence_counter[each_stem]*math.log10(N/len(postings[each_stem]))
            
        temp.append(temp_sentence)    
    return temp    
#%%
def all_file_calculate_tfidf(N,postings,doc_sentences):    
    all_idf=[]
    for docID in doc_sentences:
        all_idf.append(calculate_tfidf(N,postings,doc_sentences[docID]))
    return all_idf

#%%
def find_common_stem(sentence1,sentence2):
    list3 = set(sentence1) & set(sentence2) 
    list4 = sorted(list3, key = lambda k : sentence1.index(k))    
    return list4
#%%  
def calculate_length_dictionary(sentence_tfidf_vector):
    total=0
    for i in sentence_tfidf_vector:
        total+=sentence_tfidf_vector[i]**2
        
    return math.sqrt(total)    

        
#%%
def calculate_similarity(sentences,doc_tfidf,threshold):
    total_cos_similarity=[]
    degree=[1 for i in range(len(sentences)) ]
    for i in range(len(sentences)):        
        temp=[]
        temp.append(i)
        len1=calculate_length_dictionary(doc_tfidf[i])
        for j in range(i+1,len(sentences)):
            common_stems=find_common_stem(sentences[i], sentences[j])
            cos_sim=0
            len2=calculate_length_dictionary(doc_tfidf[j])
            for each_common_stem in common_stems:
                cos_sim+=(doc_tfidf[i][each_common_stem]*doc_tfidf[j][each_common_stem])/(len1*len2)
            
            if cos_sim > threshold:
                temp.append(j)
                degree[i]+=1
                degree[j]+=1
        
        
        total_cos_similarity.append(temp)    
                
    return total_cos_similarity,degree                
            
#%%
def all_file_calculate_similarity(doc_sentences,tfidf,threshold):
    doc_similarity=[]
    doc_degree=[]
    for i in doc_sentences:
        temp_similarity,temp_degree=calculate_similarity(doc_sentences[i],tfidf[i],threshold)  
        doc_similarity.append(temp_similarity)
        doc_degree.append(temp_degree)
    return doc_similarity,doc_degree       
    
#%% 
def power_method(cosine_matrix,n,eps):
    previous=np.ones(n)*(1/n)
    gamma=eps
    p=np.ones(n)
        
    while(gamma >= eps):
        #print(gamma," ",eps)
        p=np.dot((cosine_matrix.copy().T),previous)    
        #print(p)
        gamma=LA.norm(p-previous)
        previous=p
    
    return p

#%%  
def calculate_lexrank_score(cosine_similarity,degree,n,eps,cosine_matrix,telep_rate):
    rate=telep_rate/n
    com_telep_rate=1-telep_rate
    
    for i in range(n):        
        for j in cosine_similarity[i]:
            cosine_matrix[i][j]=1
            cosine_matrix[j][i]=1
    
    for i in range(n):
        for j in range(n):
            cosine_matrix[i][j]=rate+(cosine_matrix[i][j]/degree[i])*(com_telep_rate)
    L=power_method(cosine_matrix,n,eps)        
    
    max_value=max(L)
    L=[i/max_value for i in L]
    return L

#%%
def calculate_all_score(doc_similarity,doc_degree,eps,telep_rate):
    total_L=[]
    for i in range(len(doc_similarity)):
        n=len(doc_similarity[i])
        cosine_matrix=np.zeros((n, n))
        L=calculate_lexrank_score(doc_similarity[i],doc_degree[i],n,eps,cosine_matrix,telep_rate)
        total_L.append(L)
        
        
    return total_L    
#%%
def evaluation(collect_summary,all_gold_summary):
    rouge = Rouge()    
    average_scores={'rouge-1':{'f':0,'r':0,'p':0}, 'rouge-2':{'f':0,'r':0,'p':0} ,'rouge-l':{'f':0,'r':0,'p':0}}
    num_summary=len(collect_summary)
    for i in range(num_summary):
        scores = rouge.get_scores(collect_summary[i],all_gold_summary[i])
              
        average_scores['rouge-1']['r']+=scores[0]['rouge-1']['r']
        average_scores['rouge-1']['p']+=scores[0]['rouge-1']['p']
        average_scores['rouge-1']['f']+=scores[0]['rouge-1']['f']
        
        average_scores['rouge-2']['r']+=scores[0]['rouge-2']['r']
        average_scores['rouge-2']['p']+=scores[0]['rouge-2']['p']
        average_scores['rouge-2']['f']+=scores[0]['rouge-2']['f']
        
        average_scores['rouge-l']['r']+=scores[0]['rouge-l']['r']
        average_scores['rouge-l']['p']+=scores[0]['rouge-l']['p']
        average_scores['rouge-l']['f']+=scores[0]['rouge-l']['f']
                              
    for i in average_scores:
        for j in average_scores[i]:
            average_scores[i][j]/=num_summary           
    
    print(average_scores)     

#%%
def main(filename_docID):  
    postings={}
    doc_sentences={}

    full_file_path=sys.argv[1]
    read_file=sys.argv[2]
    stoplist=fill_stopword(stopwords_path)        
    filename_docID,N=read_all_file(full_file_path,postings,doc_sentences,stoplist)
    docID=filename_docID[read_file]
    tfidf=all_file_calculate_tfidf(N,postings,doc_sentences)
    doc_similarity,doc_degree=all_file_calculate_similarity(doc_sentences,tfidf,threshold)
    total_L=calculate_all_score(doc_similarity,doc_degree,eps,telep_rate)
    print(total_L[docID])
    
#%%
filename_docID={}
main(filename_docID)

#evaluation(collect_summary,all_gold_summary)


