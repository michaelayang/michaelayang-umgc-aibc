# michaelayang-umgc-aibc/capstone_project/deployment

This arithmetic math chatbot can be run using two methods, from the command-line:

----------------------------------------------------------------------------------

From a script setting up kubernetes:

> cd ~/michaelayang-umgc-aibc/capstone_project/deployment
> ./run_k8s.sh

Before running, please add a section such as the following to the front of k8s-manifest.yaml:

apiVersion: v1
kind: Secret
metadata:
  name: app-key-secret
type: Opaque
data:
  OPENAI_API_KEY:  [ base64-encoded API key for OpenAI, which one gets from:  echo -n '[un-encoded OPENAI_API_KEY]' | base64  ]
  GEMINI_API_KEY:  [ base64-encoded API key for Gemini ]
  APP_SECRET_KEY:  [ base64-encoded API key for app ]
  NGROK_AUTHTOKEN:  [ base64-encoded authtoken for ngrok ]

---

----------------------------------------------------------------------------------

From a raw Docker container:

> cd ~/michaelayang-umgc-aibc/capstone_project/deployment
> ./run_docker.sh

Before running, please add a section such as the following after the WORKDIR setups in Dockerfile:

# Setup necessary ENV variables
ENV OPENAI_API_KEY=[ unencoded (plain) API key for OpenAI ]
ENV GEMINI_API_KEY=[ unencoded (plain) API key for Gemini ]
ENV APP_SECRET_KEY=[ unencoded (plain) API key for app ]
ENV NGROK_AUTHTOKEN=[ unencoded (plain) authtoken for ngrok ]

----------------------------------------------------------------------------------

After starting the application, please use a web browser to go to the URL:
https://unsold-bolster-wildcat.ngrok-free.dev/login

... and login using any username desired, where that username will simply be used to tag a student's
responses in a log, to allow a teacher to match student responses to student identities.
No password is needed at this time for the first version of this software.

Then just answer the math problems asked, and make sure to also give reasoning which is even more important than
the answers to the problems.  Student answers and reasoning responses will all be logged, and can be examined 
in order to possibly produce a signficantly improved math bot "version 2.0."

----------------------------------------------------------------------------------

Teachers can create math problems for students by following the pattern in the file:
MYangAICapstoneProjectUMGCPrototypeInput.json

For instance, a new math problem file created by a teacher could be called:  MathProblems2.json
... and then the teacher must add to Dockerfile:  COPY MathProblems2.json .
... so that the math bot app will use the math problems that the teacher has created in MathProblems2.json.

