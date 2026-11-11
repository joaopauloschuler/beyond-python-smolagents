# BPSA - Beyond Python SmolAgents
# https://github.com/joaopauloschuler/beyond-python-smolagents
#
# Copyright (c) 2024-2026 Joao Paulo Schwarz Schuler and others.
# Refer to the git commit history for individual authorship.
# Licensed under the Apache License, Version 2.0

"""Agent-to-agent messages as files `from_to_id.txt` in a shared folder. Only this module knows that name format."""

import os
import re
import secrets
from contextlib import contextmanager, suppress
from typing import NamedTuple

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None


MESSAGE_TARGET_ALL = "all"
AGENT_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
AGENT_NAME_RULE = "1 to 64 characters, lowercase ASCII letters, digits or '-', not starting with '-', and not 'all'"
MESSAGE_FILE_NAME_PATTERN = re.compile(r"([a-z0-9-]+)_([a-z0-9-]+)_([0-9]{6,})\.txt")
MESSAGE_ID_DIGITS = 6
TEMPORARY_MESSAGE_FILE_SUFFIX = ".tmp"


class Message(NamedTuple):
    sender: str
    target: str
    message_id: int
    text: str


def is_valid_agent_name(name: str) -> bool:
    return isinstance(name, str) and AGENT_NAME_PATTERN.fullmatch(name) is not None and name != MESSAGE_TARGET_ALL


def check_agent_name(name: str):
    """Raises ValueError naming the rule when name is not a valid agent name."""
    if not is_valid_agent_name(name):
        raise ValueError(f"Invalid agent name {ascii(name)}: use {AGENT_NAME_RULE}.")


def check_message_target(target: str):
    """Raises ValueError when target is neither a valid agent name nor 'all'."""
    if target != MESSAGE_TARGET_ALL and not is_valid_agent_name(target):
        raise ValueError(f"Invalid message target {ascii(target)}: use 'all' or an agent name of {AGENT_NAME_RULE}.")


def message_file_name(sender: str, target: str, message_id: int) -> str:
    return f"{sender}_{target}_{message_id:0{MESSAGE_ID_DIGITS}d}.txt"


def parse_message_file_name(file_name: str) -> tuple[str, str, int] | None:
    """Returns (sender, target, id) of a message file name, or None when file_name is not exactly what message_file_name builds."""
    match = MESSAGE_FILE_NAME_PATTERN.fullmatch(file_name)
    if match is None:
        return None
    sender, target, message_id = match[1], match[2], int(match[3])
    if not is_valid_agent_name(sender) or (target != MESSAGE_TARGET_ALL and not is_valid_agent_name(target)) or message_id < 1:
        return None
    if message_file_name(sender, target, message_id) != file_name:
        return None
    return sender, target, message_id


def parsed_message_files(folder: str) -> dict[tuple[str, int], tuple[str, str]]:
    """Returns {(sender, id): (target, file name)} for the message files in folder; other names are ignored."""
    message_files = {}
    for file_name in os.listdir(folder):
        parsed = parse_message_file_name(file_name)
        if parsed is not None:
            sender, target, message_id = parsed
            message_files[(sender, message_id)] = (target, file_name)
    return message_files


def next_message_id(folder: str, sender: str) -> int:
    """Returns the highest id of sender's messages in folder, to any target, plus one; 1 when it has none."""
    return 1 + max((message_id for message_sender, message_id in parsed_message_files(folder) if message_sender == sender), default=0)


@contextmanager
def sender_lock(folder: str, sender: str):
    """Holds the exclusive flock of the file .<sender>.lock in folder, so processes of one sender take ids one at a time. No lock without fcntl."""
    if fcntl is None:
        yield
        return
    descriptor = os.open(os.path.join(folder, f".{sender}.lock"), os.O_RDWR | os.O_CREAT, 0o666)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def write_message(folder: str, sender: str, target: str, text: str) -> str:
    """Writes text as sender's next message to target in folder and returns its file name. Under sender_lock, a hard link from a temporary file
    creates the final name, so it never replaces an existing message: when that name exists, it retries with the next id."""
    check_agent_name(sender)
    check_message_target(target)
    encoded_text = text.encode("utf-8", "backslashreplace")
    temporary_path = os.path.join(folder, f".{sender}-{os.getpid()}-{secrets.token_hex(8)}{TEMPORARY_MESSAGE_FILE_SUFFIX}")
    descriptor = os.open(temporary_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with open(descriptor, "wb") as temporary_file:
            temporary_file.write(encoded_text)
        # Two targets give two names for one id, so only the lock stops two processes of one sender taking the same id.
        with sender_lock(folder, sender):
            attempted_file_name = None
            while True:
                file_name = message_file_name(sender, target, next_message_id(folder, sender))
                try:
                    os.link(temporary_path, os.path.join(folder, file_name))
                    return file_name
                except FileExistsError:
                    # A name the listing does not parse, such as another case of it, holds this id.
                    if file_name == attempted_file_name:
                        raise
                    attempted_file_name = file_name
    finally:
        with suppress(OSError):
            os.remove(temporary_path)


def message_file_text(path: str) -> str:
    """Returns the text of the message file at path, or a bracketed line naming the file and why it could not be read or decoded."""
    file_name = os.path.basename(path)
    try:
        with open(path, "rb") as message_file:
            return message_file.read().decode("utf-8")
    except UnicodeDecodeError as error:
        return f"[message file {file_name} is not valid UTF-8 at byte {error.start}]"
    except OSError as error:
        return f"[message file {file_name} could not be read: {error.strerror or type(error).__name__}]"


def read_new_messages(folder: str, reader: str, last_read_ids: dict[str, int]) -> tuple[list[Message], dict[str, int]]:
    """Returns the messages in folder to reader or 'all' from other senders with ids above last_read_ids[sender], ordered by sender and id,
    and a new dict with each sender's highest id taken. A file that cannot be read or decoded is delivered as message_file_text describes."""
    check_agent_name(reader)
    message_files = parsed_message_files(folder)
    new_ids = {}
    for sender, message_id in message_files:
        if sender != reader and message_id > last_read_ids.get(sender, 0):
            new_ids.setdefault(sender, set()).add(message_id)
    highest_new_ids = {sender: max(ids) for sender, ids in new_ids.items()}
    # A listing taken during a write may miss an id that existed below a listed one; a second listing then holds it.
    if any(len(ids) != highest_new_ids[sender] - last_read_ids.get(sender, 0) for sender, ids in new_ids.items()):
        message_files = parsed_message_files(folder) | message_files
    updated_last_read_ids = dict(last_read_ids) | highest_new_ids
    messages = []
    for (sender, message_id), (target, file_name) in sorted(message_files.items()):
        if sender in highest_new_ids and last_read_ids.get(sender, 0) < message_id <= highest_new_ids[sender] and target in (reader, MESSAGE_TARGET_ALL):
            messages.append(Message(sender, target, message_id, message_file_text(os.path.join(folder, file_name))))
    return messages, updated_last_read_ids
