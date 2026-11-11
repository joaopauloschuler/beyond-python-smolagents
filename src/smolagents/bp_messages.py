# BPSA - Beyond Python SmolAgents
# https://github.com/joaopauloschuler/beyond-python-smolagents
#
# Copyright (c) 2024-2026 Joao Paulo Schwarz Schuler and others.
# Refer to the git commit history for individual authorship.
# Licensed under the Apache License, Version 2.0

"""Agent-to-agent messages as files `from_to_id.txt` in a shared folder. Only this module knows that name format."""

import argparse
import json
import os
import re
import secrets
import sys
import time
from contextlib import contextmanager, suppress
from typing import NamedTuple

from smolagents.bp_utils import get_env, replace_file_with_text, terminal_display_escapes, visible_encodable_text

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
WAIT_POLL_SECONDS = 0.5
DEFAULT_WAIT_TIMEOUT_SECONDS = 300.0
EXIT_ERROR = 1
EXIT_WAIT_TIMEOUT = 3
EXIT_INTERRUPTED = 130
MESSAGE_BODY_LINE_PREFIX = "|"

_warned_last_read_paths: set[str] = set()


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


def stderr_literal(text: str) -> str:
    """Returns text as one literal line for stderr: visible_encodable_text for its encoding, with newline escaped as well."""
    return visible_encodable_text(text, sys.stderr.encoding or "utf-8").replace("\n", "\\n")


def last_read_file_path(folder: str, reader: str) -> str:
    """Returns the path of the file .<reader>.read.json in folder, which holds reader's last-read ids; its name never parses as a message."""
    check_agent_name(reader)
    return os.path.join(folder, f".{reader}.read.json")


def is_last_read_ids(value) -> bool:
    return isinstance(value, dict) and all(is_valid_agent_name(sender) and type(message_id) is int and message_id >= 0 for sender, message_id in value.items())


def load_last_read_ids(folder: str, reader: str) -> dict[str, int]:
    """Returns {sender: last id read} of reader in folder: empty when the file is missing, and empty with one stderr line per process and path
    when it is not a JSON object of agent names to ids. Raises OSError when the file exists but cannot be read."""
    path = last_read_file_path(folder, reader)
    try:
        with open(path, "rb") as last_read_file:
            content = last_read_file.read()
    except FileNotFoundError:
        return {}
    try:
        last_read_ids = json.loads(content.decode("utf-8"))
    except (ValueError, RecursionError):
        last_read_ids = None
    if is_last_read_ids(last_read_ids):
        return last_read_ids
    if path not in _warned_last_read_paths:
        _warned_last_read_paths.add(path)
        print(f"Ignoring {stderr_literal(path)}: not a JSON object of agent names to message ids; "
              f"every message in the folder counts as unread.", file=sys.stderr)
    return {}


def save_last_read_ids(folder: str, reader: str, last_read_ids: dict[str, int]):
    """Writes last_read_ids as reader's last-read ids in folder through replace_file_with_text (a temporary file renamed over the old one)."""
    replace_file_with_text(last_read_file_path(folder, reader), json.dumps(last_read_ids, sort_keys=True) + "\n")


def receive_new_messages(folder: str, reader: str) -> list[Message]:
    """Returns reader's new messages in folder and records them as read. It saves after reading, so a crash in between delivers them again rather than losing them."""
    last_read_ids = load_last_read_ids(folder, reader)
    messages, updated_last_read_ids = read_new_messages(folder, reader, last_read_ids)
    if updated_last_read_ids != last_read_ids:
        save_last_read_ids(folder, reader, updated_last_read_ids)
    return messages


def printable_messages(messages: list[Message], encoding: str) -> str:
    """Returns messages as plain text for a person or a model: a header and an end line per message, and every text line after a '|' prefix,
    so a text cannot forge either line. visible_encodable_text leaves newline as the only line break."""
    blocks = []
    for message in messages:
        body_lines = visible_encodable_text(message.text, encoding).split("\n")
        blocks.append(f"=== message {message.message_id} from {message.sender} to {message.target} ===\n"
                      + "".join(f"{MESSAGE_BODY_LINE_PREFIX} {line}\n" if line else f"{MESSAGE_BODY_LINE_PREFIX}\n" for line in body_lines)
                      + f"=== end of message {message.message_id} from {message.sender} ===\n")
    return "".join(blocks)


def print_and_record_new_messages(folder: str, reader: str, is_peek: bool) -> int:
    """Prints reader's new messages in folder and, unless is_peek, records them as read after printing. Returns how many it printed."""
    last_read_ids = load_last_read_ids(folder, reader)
    messages, updated_last_read_ids = read_new_messages(folder, reader, last_read_ids)
    if messages:
        sys.stdout.write(printable_messages(messages, sys.stdout.encoding or "utf-8"))
        sys.stdout.flush()
    if not is_peek and updated_last_read_ids != last_read_ids:
        save_last_read_ids(folder, reader, updated_last_read_ids)
    return len(messages)


def default_message_folder() -> str | None:
    """Returns the folder of BPSA_MESSAGE_FOLDERS when it names exactly one (one per line, as BPSA_MCP); otherwise None."""
    folders = [line.strip() for line in (get_env("BPSA_MESSAGE_FOLDERS") or "").splitlines() if line.strip()]
    return folders[0] if len(folders) == 1 else None


def message_command_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--folder", help="message folder; default: BPSA_MESSAGE_FOLDERS when it names exactly one folder")
    common.add_argument("--name", help="your agent name; default: BPSA_AGENT_NAME")
    parser = argparse.ArgumentParser(prog="python -m smolagents.bp_messages", description="Send and read agent-to-agent messages in a message folder.")
    commands = parser.add_subparsers(dest="command", required=True)
    send = commands.add_parser("send", parents=[common], help="write one message and print its file name")
    send.add_argument("--to", required=True, help="an agent name or 'all'")
    send.add_argument("text", nargs="?", default="-", help="message text; '-' or absent reads it from stdin")
    new = commands.add_parser("new", parents=[common], help="print the messages you have not read and record them as read")
    wait = commands.add_parser("wait", parents=[common], help="as new, but first wait until at least one message arrives")
    wait.add_argument("--timeout", type=float, default=DEFAULT_WAIT_TIMEOUT_SECONDS,
                      help=f"seconds to wait (default {DEFAULT_WAIT_TIMEOUT_SECONDS:g}); on timeout exit code {EXIT_WAIT_TIMEOUT} with nothing printed")
    for reading_command in (new, wait):
        reading_command.add_argument("--peek", action="store_true", help="print without recording the messages as read")
    return parser


def run_message_command(args: argparse.Namespace) -> int:
    folder = args.folder if args.folder is not None else default_message_folder()
    if folder is None:
        raise ValueError("--folder is required when BPSA_MESSAGE_FOLDERS does not name exactly one folder.")
    name = args.name if args.name is not None else get_env("BPSA_AGENT_NAME")
    if name is None:
        raise ValueError("--name is required when BPSA_AGENT_NAME is unset.")
    check_agent_name(name)
    if not os.path.isdir(folder):
        raise ValueError(f"The message folder {folder} does not exist or is not a folder.")
    if args.command == "send":
        check_message_target(args.to)
        text = sys.stdin.buffer.read().decode("utf-8", "surrogateescape") if args.text == "-" else args.text
        print(write_message(folder, name, args.to, text))
        return 0
    if args.command == "new":
        print_and_record_new_messages(folder, name, args.peek)
        return 0
    if not args.timeout >= 0:
        raise ValueError("--timeout must be a number of seconds, 0 or more.")
    terminal_display_escapes()  # builds the escape table (about 1 s) before the first message, not after it
    deadline = time.monotonic() + args.timeout
    while print_and_record_new_messages(folder, name, args.peek) == 0:
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            return EXIT_WAIT_TIMEOUT
        time.sleep(min(WAIT_POLL_SECONDS, remaining_seconds))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Runs the send, new and wait commands. An error prints one line on stderr and returns EXIT_ERROR; Ctrl-C returns EXIT_INTERRUPTED."""
    args = message_command_parser().parse_args(argv)
    try:
        return run_message_command(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return EXIT_INTERRUPTED
    except (ValueError, OSError) as error:
        print(f"Error: {stderr_literal(str(error))}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
