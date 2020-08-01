import logging

from RenderEngine.GenerateNotification import server

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    server()
