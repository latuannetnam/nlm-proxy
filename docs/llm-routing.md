# Change openai proxy flow
##  Smart request routing using LLM for OpenAI proxy:
 - Receive request from client 
 - Use LLM to quickly check request and category in different request types:
    + If request about information from NotebookLM: NotebookLM
    + If request not directly about NotebookLM, for example: summary conversation history, generate follow up question etc -> return LLM task

-  If request type = LLM task -> use OpenAI-compatible endpoint to ask LLM to do task then return to client. Using same configuration mechanism to configure OpenAI endpoint (URL, API key, model etc). The response from LLM using the same method of openai proxy to stream response and reasoning result to client

 - If request type = NotebookLM  
   + query list of available notebooks (the list can be customized, or default using all), get summary of each notebook then cache into memory for reuse
   + Use LLM from OpenAI-compatible endpoint to use the notebook summary to quickly check which notebook_id possible have the answer for request -> return notebook__id  
   + use original flow to send request to NotebookLM with notebook_id and return directly to client

## Other requirements
 - Currently openai proxy return model list based on list of notebook. To support smart request routing, add special model to the model list, so client will use this model to utilize smart routing flow. Other notebooklm models processing flow will remain the same as before
 - Use prompt template file to customize LLM processing task
 - The list of notebook from NotebookLM can be config, default=all
 

