# set base image (host OS)
FROM python:3.9
# cannot use the slim image because of missing dependencies for pycocotools

# set the working directory in the container
WORKDIR /code

# Create new user with UID
RUN adduser --disabled-password --gecos '' --system --uid 1001 python && chown -R python /code

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# copy the content of the local src directory to the working directory
COPY . ./

# Set user to newly created user
USER 1001

# command to run on container start
CMD [ "python", "main.py" ]