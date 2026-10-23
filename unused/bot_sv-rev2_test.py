import requests
import threading
import uuid
import time
from flask import Flask, render_template, jsonify, request, redirect, url_for, abort
# from test_first_agent_bak import get_agent_ans
# from test_first_agent_aa import get_agent_ans,define_user_needs
from top_LLM import process_query
app = Flask(__name__)

# Global storage for processing results
processing_results = {}

def process_agent_query(request_id, userid, question,roles):
    """
    Background function to process agent query
    """
    callback_url = "https://hxwlzzxh-48080.asse.devtunnels.ms/llm_result" 
    ""
    # callback_url = "http://192.168.10.8/llm_result"
    
    try:
        print(f"🔄 Processing request {request_id} for user {userid}")
        
        # Process the agent query
        # mode = define_user_needs(question)

        # print(mode)

        # result, tool_result = get_agent_ans(userid, question+" ใช้ tool ="+mode)
        # result, tool_result = get_agent_ans(userid, question)
        print(roles)
        permiss = 'Group Permission: \n'
        group_map = {
            'AI-DT': 'All sales',
            'AI-DB1': 'Sales in DB1',
            'AI-DB2': 'Sales in DB2',
            'AI-DB3': 'Sales in DB3',
            'AI-IB': 'Sales in IB',
        }
        # print(group_map.get('AI-DT'))
        # print(group_map.get('AI-DB1'))
        # print("this is a test")
        module_access= ''
        if roles:
            for role in roles:
                if 'AI' in role:
                    print(group_map.get(role))
                    print('###########')
                    module_access = module_access + group_map.get(role,'')+'\n'
                    permiss = permiss +role +'\n'
                    # print(permiss)
                    # print('this is a role')
                    # print(roles)
            result,tool_result,mode = permiss+"##############\nmodule access: \n"+module_access,permiss+"##############\nmodule access: \n"+module_access,1
        else:
            result,tool_result,mode = 'คุณไม่มีสิทธิ์ กรุุณา LOGIN','คุณไม่มีสิทธิ์ กรุุณา LOGIN',3

        if mode == 2:
            print("Phone search mode detected, skipping tool result posting")

            result_data = {
                'request_id': request_id,
                'user_id': userid,
                'original_query': question,
                'status': 'completed',
                'mode': mode,
                'search_type': 'phone_search',
                'data': {
                    'answer': str(result),
                    'phone_data': tool_result  # Different key for phone data
                },
                'timestamp': time.time()
            }
        # print('#######################')
        # print(tool_result)
        # print('#######################')
        # Prepare data for posting
        elif mode ==3:
            result_data = {
                'request_id': request_id,
                'user_id': userid,
                'original_query': question,
                'status': 'completed',
                'mode': mode,
                'search_type': '',
                'data': {
                    'answer': str(result),
                    'phone_data': tool_result  # Different key for phone data
                },
                'timestamp': time.time()
            }
        # prin
        else:
            result_data = {
                'request_id': request_id,
                'user_id': userid,
                'original_query': question,
                'mode': mode,
                'status': 'completed',
                'data': {
                    'answer': result,
                    'all_data': {'link':tool_result}
                },
                'timestamp': time.time()
            }
        
        # Store the result locally (backup)
        processing_results[request_id] = result_data
        
        # POST to your endpoint
        try:
            print(result_data)
            test_data = 'xxxxxxxxxx'
            response = requests.post(callback_url, json=result_data, timeout=5)
            
            if response.status_code == 200:
                print(f"✅ Successfully posted result to {callback_url}")
                print(f"✅ Response: {response.text}")
            else:
                print(f"❌ Failed to post result: HTTP {response.status_code}")
                print(f"❌ Response: {response.text}")
                
        except requests.exceptions.RequestException as post_error:
            result_data = {
                'request_id': request_id,
                'user_id': userid,
                'timestamp': time.time()
            }
            response = requests.post(callback_url, json=result_data, timeout=5)
            print(f"❌ Network error posting to {callback_url}: {post_error}")
        
        print(f"✅ Completed request {request_id}")
        
    except Exception as e:
        print(f"❌ Error processing request {request_id}: {e}")
        
        # Prepare error data
        error_data = {
            'request_id': request_id,
            'user_id': userid,
            'original_query': question,
            'status': 'error',
            'error': str(e),
            'timestamp': time.time()
        }
        
        # Store error locally
        processing_results[request_id] = error_data
        
        # POST error to your endpoint
        try:
            response = requests.post(callback_url, json=error_data, timeout=30)
            
            if response.status_code == 200:
                print(f"✅ Successfully posted error to {callback_url}")
            else:
                print(f"❌ Failed to post error: HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as post_error:
            print(f"❌ Network error posting error to {callback_url}: {post_error}")

@app.route('/post_ibot_query', methods=['POST'])
def post_ibot_query():
    """
    Submit query for processing - returns immediately
    """
    data = request.get_json()
    print(data)
    question = data.get('query', '')
    print('test')
    print(question)
    userid = data.get('userId', '')
    roles = data.get('roles','')
    print(roles)
    if question:
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Mark as processing
        processing_results[request_id] = {
            'status': 'processing',
            'timestamp': time.time()
        }
        
        # Start background processing
        thread = threading.Thread(
            target=process_agent_query, 
            args=(request_id, userid, question,roles)
        )
        thread.daemon = True
        thread.start()
        
        print(f"🚀 Started processing request {request_id}")
        
        # Return immediately with request ID
        return jsonify({
            'status': 200, 
            'message': 'Processing started successfully',
            'request_id': request_id,
            'check_url': f'/get_result/{request_id}'
        })
    else:
        return jsonify({'status': 202, 'msg': 'No query found'})


if __name__ == '__main__':
    app.run(port=4001, threaded=True,host='0.0.0.0')