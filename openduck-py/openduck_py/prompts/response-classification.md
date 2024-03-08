You are parsing the transcription of speech recorded by an interactive voice agent.
You should classify the intent and content of the following transcription:

{{transcription}}

The intent should correspond to an intent from the list ["stop", null]. "stop" should correspond to the intent of the user to stop the conversation. "name" should correspond to the user saying their name. "other" should refer to anything else. For example, "well, actually, stop" should be classified as "stop", but "stop and shop is my favorite store" would be null.

The content should correspond to information from the user's chat that is useful to record about them.
Right now the only information to record should be ["name"]. For example, "My name is Sam" would correspond to an content {"name":"Sam"}.

Please return in the format {"intent": INTENT, "content": {"name": NAME}}. So, for example you could return

{"intent": null, "content": {"name": null}}
{"intent": null, "content": {"name": "Sam"}}
{"intent": "stop", "content": {"name": "Sam"}}
