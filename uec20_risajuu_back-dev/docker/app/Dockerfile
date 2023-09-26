FROM nvcr.io/nvidia/pytorch:23.04-py3

ARG USER_ID
ARG GROUP_ID
ARG USER_NAME
ARG GROUP_NAME
USER root

# RUN groupadd -g ${GROUP_ID} ${GROUP_NAME} \
#     && useradd -u ${USER_ID} -g ${GROUP_ID} -m ${USER_NAME}
# USER ${USER_NAME}


RUN apt-get update
RUN apt-get -y install locales && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV TERM xterm

COPY requirements.txt /root/

RUN apt-get install -y vim less
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

# RUN apt-get install -y portaudio19-dev
#RUN pip install --global-option='build_ext' --global-option='-I/usr/local/Cellar/portaudio/19.7.0/include' --global-option='-L/usr/local/Cellar/portaudio/19.7.0/lib' pyaudio
# RUN apt-get update && apt-get install -y alsa-base alsa-utils
RUN pip install -r /root/requirements.txt
RUN pip install fugashi ipadic
# RUN python -m pip install git+https://github.com/openai/whisper.git
RUN apt-get -y install fonts-ipafont-gothic

# RUN groupadd -g ${GROUP_ID} ${GROUP_NAME} \
#     && useradd -u ${USER_ID} -g ${GROUP_ID} -m ${USER_NAME}
# USER ${USER_NAME}

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all
# ENV PYTHONPATH "${PYTHONPATH}:/home/naoki/vr_voice_chat/vr_voice_chat/"
# ENV PYTHONPATH "${PYTHONPATH}:/root/app/"

# fastapi のサーバーを立ち上げる
#CMD ["uvicorn", "main:app", "--reload", "--host", "127.0.0.1", "--port", "8000"]