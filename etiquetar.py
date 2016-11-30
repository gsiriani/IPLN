# -*- encoding: utf-8 -*-

import pandas
import nltk

corpus = pandas.read_csv("corpus_pj.csv", delimiter=',', skip_blank_lines=True, encoding='utf-8')
corpus = corpus.drop_duplicates()
sentencias = corpus["sentencia"]


import sys
import json

#sys.path.append('/home/kovy/IPLN/FreeLing/APIs/python')
import freeling
from codecs import open, BOM_UTF8
import re

# Configuracion de freeling
# -------------------------
FREELINGDIR = "/usr/share";
DATA = FREELINGDIR+"/freeling/"
LANG = "es"

freeling.util_init_locale("default")

op = freeling.maco_options("es")
#op.set_active_modules(1,1,1,1,1,1,1,1,1,1,0)
op.set_data_files( "", 
                   DATA + "common/punct.dat",
                   DATA + LANG + "/dicc.src",
                   DATA + LANG + "/afixos.dat",
                   "",
                   DATA + LANG + "/locucions.dat", 
                   DATA + LANG + "/np.dat",
                   DATA + LANG + "/quantities.dat",
                   DATA + LANG + "/probabilitats.dat")
op.set_retok_contractions(False)

# lg  = freeling.lang_ident(DATA+"common/lang_ident/ident-few.dat")
mf  = freeling.maco(op)

tk  = freeling.tokenizer(DATA+LANG+"/tokenizer.dat")
sp  = freeling.splitter(DATA+LANG+"/splitter.dat")
tg  = freeling.hmm_tagger(DATA+LANG+"/tagger.dat",True,2)
tagset = freeling.tagset(DATA+LANG+"/tagset.dat")
ner = freeling.ner(DATA+LANG+"/nerc/ner/ner-ab-poor1.dat")
nec = freeling.nec(DATA+LANG+"/nerc/nec/nec-ab-poor1.dat")

# sen = freeling.senses(DATA+LANG+"/senses.dat");
# ukb = freeling.ukb(DATA+LANG+"/ukb.dat")



# Función que etiqueta las palabras de una oración 
def tag (sent):
    out = {}
    # l = tk.tokenize(sent)
    # ls = sp.split(l) # old value 0
    ls = sent
    ls = mf.analyze(ls)
    ls = tg.analyze(ls)
    ls = ner.analyze(ls)
    ls = nec.analyze(ls)
    
    wss = []
    wss_aux = []
    in_contr = False
    ws = ls.get_words()
    indice_original = 0
    for w in ws:
        # Obtengo el analisis
        an = w.get_analysis()
        a = an[0]
        wse = dict(wordform =  w.get_form(),
                   lemma = a.get_lemma(),
                   tag = a.get_tag(),
                   prob = a.get_prob(),
                   analysis = len(an),
                    sent=a,ls=ls, palabras = 0)
        
        # Chequeo si la palabra es parte de una contraccion
        if (indice_original < len(sent)) and ((w.get_form() == sent[indice_original].get_form()) or (sent[indice_original].get_form() in w.get_form())):
            if (sent[indice_original].get_form() in w.get_form()):
                # Palabra compuesta
                contador_palabra = 0
                wss_comp_aux = []
                while (indice_original < len(sent)) and (sent[indice_original].get_form() in w.get_form()):
                    if (contador_palabra==0):
                        wse_comp = dict(wordform =  sent[indice_original].get_form(),
                               lemma = a.get_lemma(),
                               tag = a.get_tag(),
                               prob = a.get_prob(),
                               analysis = len(an),
                               sent=a,ls=ls, palabras = 0)
                    else:
                        wse_comp = dict(wordform =  sent[indice_original].get_form(),
                               lemma = a.get_lemma(),
                               tag = a.get_tag() + "I",
                               prob = a.get_prob(),
                               analysis = len(an),
                               sent=a,ls=ls, palabras = 0)
                    contador_palabra+=1
                    wss_comp_aux.append(wse_comp)
                    indice_original += 1
                indice_original += -1 # corrijo
                
            indice_original += 1 
            if len(wss_comp_aux)>1:
                    #si es una palabra compuesta
                    for w in wss_comp_aux:
                        wss.append(w)
                        
            if in_contr:
                # Termino la contraccion
                wss.append(wcont)
                wss += wss_aux
                wss_aux = []
                in_contr = False
                
            if len(wss_comp_aux)==1:
                #si no es una palabra compuesta la agrego
                wss.append(wse)
            
        else:
            if (indice_original < len(sent)):
                if not in_contr:
                    # Palabra original
                    if sent[indice_original].get_form() in ['al', 'del']: # contracciones buscadas
                        wcont = dict(wordform =  sent[indice_original].get_form(),
                                     lemma = '_',
                                     tag = '_',
                                     prob = 0,
                                     analysis = 0,
                                     sent=sent[indice_original].get_form(),ls='_', palabras = 1)                  
                        indice_original += 1     
                        in_contr = True

                        # Palabra actual
                        wss_aux.append(wse)
                    else: # asumo que la palabra encontrada fue agregada por freeling
                        wss.append(wse)

                else:                
                    wcont['palabras'] += 1
                    wss_aux.append(wse)
            
            

        
    return wss


# Mapeo de etiquetas entre la salida de freeling y UPOSTAG
tagMapping = [("NC","NOUN"),("NP","PROPN"),("A","ADJ"), ("PP","PRON"),("D","DET"),("Z","NUM"),("W","ADJ"),("VM","VERB")
              ,("V","AUX"),("R","ADV"),("CC","CONJ"),("CS","SCONJ"),("I","INTJ"),("F","PUNCT"),("SP","ADP"),("_","_")]

# Función que mapea las etiquetas de freeling con las de UPOSTAG
def tagToStr(tag):
    r = [s for t,s in tagMapping if tag.startswith(t)]
    return "??" + tag + "??" if len(r) == 0 else r[0]

# Función que convierte la etiqueta en IOB para entidades con nombre
def tagToBIO(tag):
    if tag == "NP00SP0":
        return "B-PER"
    elif tag == "NP00G00":
        return "B-LOC"
    elif tag == "NP00O00":
        return "B-ORG"
    elif tag == "NP00V00":
        return "B-OTR"
    else:
        return "O"

def getLine(sentencia):
    return "{:22}\t{:25}\t{:8}\t{:50}\t_\t_\t_\t_\t{}".format(sentencia['wordform'].encode('utf-8'), 
                                            sentencia['lemma'].encode('utf-8'), 
                                            tagToStr(tagset.get_short_tag(sentencia['tag'])) if sentencia['tag'] != '_' else '_', 
                                            tagset.get_msd_string(sentencia['tag']) if sentencia['tag'] != '_' else '_',
                                            tagToBIO(sentencia['tag'])
                                           )

def getLines(oracion):
    i = 1
    salida = ''
    for palabra in oracion:
        if palabra['palabras'] == 0:
            salida += '\n' + str(i) + '\t' + getLine(palabra)
            i += 1
        else:
            salida += '\n' + str(i) + '-' + str(i + palabra['palabras'] -1) + '\t' + getLine(palabra)
    return salida
    #return "\n".join([str(i) + "\t" + getLine(s) for i, s in enumerate(oracion, start=1)])



corpus_taggeado = []
for i, s in enumerate(sentencias[0:1]):       
    corpus_taggeado.append("# " + str(i))
    l = tk.tokenize(s)
    ls = sp.split(l) # old value 0
    for oracion in ls:
        tagged = tag(oracion)
        corpus_taggeado.append(getLines(tagged))
        corpus_taggeado.append("\n")

        
open("sentencias_etiquetadas_0-1", "w").write(BOM_UTF8 + "\n".join(corpus_taggeado))
