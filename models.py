#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 01:21:02 PM EST 2023 
author: Ryan Hildebrandt, github.com/ryancahildebrandt
"""
# imports
import sentence_transformers as st
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import random
random.seed(42)

# st models
mdl_robertav3 = st.SentenceTransformer("flax-sentence-embeddings/all_datasets_v3_roberta-large")
mdl_micse = st.SentenceTransformer("sap-ai-research/miCSE")
mdl_sentencet5 = st.SentenceTransformer("sentence-transformers/sentence-t5-base")
mdl_robertav1 = st.SentenceTransformer("sentence-transformers/all-roberta-large-v1")
mdl_distilrobertav1 = st.SentenceTransformer("sentence-transformers/all-distilroberta-v1")
mdl_mpnetv2 = st.SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
mdl_minilml12 = st.SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
mdl_minilml6 = st.SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

mdl_list = [i for i in dir() if "mdl_" in i]

# bow vocabs
vocab_ng = [
    #alt.atheism
    "higher", "religion", "believe", "atheist", "atheism", "prove", "evidence", "existence", "morality", "belief",
    #comp.graphics
    "graphics", "image", "resolution", "jpeg", "png", "picture", "color", "gif", "screen", "display",
    #comp.os.ms-windows.misc
    "ms", "office", "pc", "os", "product", "software", "desktop", "folder", "disk", "system",
    #comp.sys.ibm.pc.hardware
    "machine", "drive", "hardware", "computer", "usb", "bus", "motherboard", "floppy", "dos", "port",
    #comp.sys.mac.hardware
    "apple", "mac", "unix", "touchpad", "disc", "store", "ios", "operating", "card", "monitor",
    #comp.windows.x
    "windows", "file", "development", "run", "microsoft", "personal", "code", "program", "server", "application",
    #misc.forsale
    "$", "price", "dollars", "new", "like", "condition", "sale", "sell", "contact", "offer",
    #rec.autos
    "car", "auto", "vehicle", "engine", "mileage", "used", "driver", "brake", "oil", "gas",
    #rec.motorcycles
    "bike", "motorcycle", "ride", "wheel", "tire", "helmet", "bikes", "riding", "handle", "cc",
    #rec.sport.baseball
    "base", "ball", "baseball", "inning", "homerun", "strike", "bat", "hit", "runs", "pitch",
    #rec.sport.hockey
    "hockey", "puck", "stick", "skate", "period", "nhl", "ice", "goalie", "shot", "goal",
    #sci.crypt
    "encryption", "security", "key", "public", "private", "access", "information", "hash", "encode", "decode",
    #sci.electronics
    "ground", "voltage", "power", "wire", "circuit", "wiring", "electrical", "amp", "connect", "battery",
    #sci.med
    "medical", "hospital", "doctor", "nurse", "care", "patient", "disease", "treatment", "recovery", "research",
    #sci.space
    "space", "atmosphere", "earth", "nasa", "moon", "lunar", "rocket", "satellite", "solar", "orbit",
    #soc.religion.christian
    "god", "jesus", "church", "bible", "sacrement", "catholic", "protestant", "faith", "lord", "christ",
    #talk.politics.guns
    "ammendment", "2nd", "second", "gun", "ammunition", "firearm", "weapon", "crime", "killed", "shooting",
    #talk.politics.mideast
    "middle", "east", "arms", "conflict", "fighting", "region", "control", "war", "state", "regime",
    #talk.politics.misc
    "government", "president", "states", "federal", "national", "party", "jobs", "debate", "race", "issue",
    #talk.religion.misc
    "religious", "soul", "christian", "jewish", "muslim", "hindu", "buddhist", "moral", "commandment", "sect",
    ]

vocab_bt = [
    # check_refund_policy
    "refund","policy","return",
    # complaint
    "file","complaint","report",
    # get_invoice
    "invoice","charge","bill",
    # newsletter_subscription
    "newsletter","subscription","subscribe",
    # place_order
    "place","purchase","new",
    # delivery_options
    "delivery","date","address",
    # cancel_order
    "cancel","recent","placed",
    # contact_customer_service
    "contact","customer","service",
    # create_account
    "account","register","create",
    # check_cancellation_fee
    "cancellation","fee","penalty",
    # registration_problems
    "sign","problems","signing",
    # check_payment_methods
    "credit","method","card",
    # edit_account
    "updating","edit","profile",
    # contact_human_agent
    "agent","representative","person",
    # set_up_shipping_address
    "shipping","set","up",
    # get_refund
    "reimbursement","money","back",
    # delete_account
    "delete","close","remove",
    # recover_password
    "password","recover","forgot",
    # change_order
    "update","order","details",
    # review
    "review","feedback","comment",
    # delivery_period
    "arrive","period","deliver",
    # change_shipping_address
    "changes","modify","different",
    # payment_issue
    "payment","problem","issue",
    # track_order
    "track","number","tracking",
    # switch_account
    "switch","change","other",
    # check_invoices
    "invoices","view","see",
    # track_refund
    "processed","status","processing"
    ]
