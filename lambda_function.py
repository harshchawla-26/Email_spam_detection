# Imports
import json
import urllib.parse # urllib: Package for url handling in python, parse: module that provides functions for manipulating urls
import boto3 # AWS SDK that allows connection between AWS services 
from email.parser import BytesParser
from email import policy
import email # library for managing email messages 
import os # library to work with environment variables 
from utils import one_hot_encode # 'utils' is our own python file that is uploaded into this function and contains machine learning preprocessing functions
from utils import vectorize_sequences


# Defining constants 
# ENDPOINT_NAME = os.environ.get('SAGEMAKERENDPOINT') # Environment variable defined in cloud formation 
# AWS_REGION = os.environ.get('REGION') # Environment variable defined in cloud formation 
ENDPOINT_NAME = 'sms-spam-classifier-mxnet-2022-11-27-19-50-29-196'
AWS_REGION = 'us-east-1'
VOCABULARY_LENGTH = 9013 # Used in our ML notebook 
CHARSET = "UTF-8" # Encoding 


# Creating clients
s3 = boto3.client('s3') # Creating an s3 client 
runtime = boto3.Session().client(service_name = 'sagemaker-runtime', region_name = AWS_REGION) # Creating a runtime for sagemaker 
ses_client = boto3.client('ses',region_name = AWS_REGION) # Creating ses client


# Handler is the function that is run automatically when lambda is invoked, like the main function
def lambda_handler(event, context): # Function that works on the email extracted from s3 bucket 


    bucket = event['Records'][0]['s3'] # Getting the bucket from the event 
    key = bucket['object']['key'] # Getting the key of email just added to the bucket 
    key = urllib.parse.unquote_plus(key, encoding='utf-8') # Formatting key correctly by converting to string and unquoting it
    bucket_name = bucket['bucket']['name'] # Extracting bucket name 
    

    try:
        response = s3.get_object(Bucket = bucket_name, Key = key) # Getting object that was added into the s3 bucket
        response_body = response['Body'].read() # Extracting just the body from the whole response 
        email_raw_msg = BytesParser(policy = policy.SMTP).parsebytes(response_body) # The response body is bytes so we parse it
        # print("email_raw_msg: {}".format(email_raw_msg))
       

        email_datetime = email_raw_msg['Date'] # Getting email date
        email_subject = email_raw_msg['Subject'] # Getting email subject
        from_email = email_raw_msg['From'] # Getting sender of email
        to_email = email_raw_msg['To'] # Getting receiver of email
        email_body = email_raw_msg.get_body(preferencelist = ('plain')) # Getting message body in plain text format 
        email_body = ''.join(email_body.get_content().splitlines(keepends = True))
        if email_body == None:
            email_body = ''
        else:
            email_body = email_body
        #email_body = '' if email_body == None else email_body
   

        input_mail = [email_body.strip()] # Stripping email body of the "\n"
        one_hot_test_messages = one_hot_encode(input_mail, VOCABULARY_LENGTH) # Encoding our body since the ML model needs input in this format 
        encoded_test_messages = vectorize_sequences(one_hot_test_messages, VOCABULARY_LENGTH) 
        data = json.dumps(encoded_test_messages.tolist()) # Converting encoded body to list since model requires it 
       

        sagemaker_response = runtime.invoke_endpoint(EndpointName = ENDPOINT_NAME, ContentType = 'application/json', Body = data) # Passing email to sagemaker notebook 
        raw_sagemaker_response = sagemaker_response['Body'].read().decode() # Extracting decoded body of the sagemaker response 
        raw_sagemaker_response = json.loads(raw_sagemaker_response) # Loading the body into json format 
        label = raw_sagemaker_response['predicted_label'][0][0] # Extracting label of prediction
        predicted_probability = raw_sagemaker_response['predicted_probability'][0][0] # Extracting probability of classification

       
        if int(label) == 1: # Converting classification to text classes
            classification_label = "SPAM"
        else:
            classification_label = "HAM"
       

        reply_message = construct_reply_message(email_datetime, email_subject, classification_label, predicted_probability, email_body) # Constructing a reply email with predictions 
        reply_message_subject = "Spam Detection Report" # Subject of reply email 


        response = ses_client.send_email( # Sending email through ses 
        Destination={
            'ToAddresses': [
                str(from_email),
            ],
        },
        Message={
            'Body': {
                'Text': {
                    'Charset': CHARSET,
                    'Data': reply_message,
                },
            },
            'Subject': {
                'Charset': CHARSET,
                'Data': reply_message_subject,
            },
        },
        Source=str(to_email),
    )
        return reply_message

    except Exception as e: # Raising errors 
        print('Error: '.format(str(e)))
        raise e


# Function that creates reply email
def construct_reply_message(email_receive_date, subject, classification_label, probability_score, email_body): 


    message_statement_one = "We received your email sent at {} with the subject {}.".format(email_receive_date, subject) # Line 1 of reply email  


    email_body_len = len(email_body) # Calculating the length of the email body. We want to display max 240 characters of the OG email sent 
    if email_body_len > 240:
        email_body = email_body[:240] # Stripping string to only the first 240 characters 
    message_statement_two = "Here is a {} character sample of the email body:".format(len(email_body)) # Line 2 of reply email  


    message_statement_three = "The email was categorized as {} with a {}% confidence.".format(classification_label, round(probability_score*100,4)) # Line 3 of reply email  
    

    final_message = message_statement_one + "\n" + message_statement_two + "\n" + email_body + "\n" + message_statement_three # Concatenation all lines to create final message 


    return final_message