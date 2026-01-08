# backend/train_control.py

def send_train_stop_command(train_id, lat, lon):
    """
    Train signalling system ko STOP command bhejna
    jab emergency situation detect ho.
    """
    print(
        f"🚆 TRAIN STOP COMMAND SENT → "
        f"Train: {train_id}, Location: ({lat}, {lon})"
    )