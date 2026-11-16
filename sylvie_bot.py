#!/usr/bin/env python3
"""
Discord bot that chats using the Sylvie LLM model in model.pt (uses the chat special tokens
and system prompt from sylvie_llm.py; works best after Stage 2 instruction fine-tuning).

Setup:
    pip install discord.py torch
    set DISCORD_TOKEN=your_token        (Windows)   /   export DISCORD_TOKEN=your_token  (Linux/Mac)
    python sylvie_bot.py

Keep this file in the same folder as sylvie_llm.py and model.pt.

Commands:
    /chat <message>     or   !chat <message>      talk to the model
    /resetchat          or   !resetchat           clear your chat history
    @BotName <message>                            also works (mention the bot)
"""
import os
import sys
import asyncio

import discord
from discord import app_commands
from discord.ext import commands

from sylvie_llm import App, chat_prompt_ids, with_system  # tokenizer, model, settings, chat template

MODEL_PATH = "model.pt"
MAX_TURNS = 8  # remembered (user, bot) exchanges per user
LOG_CHANNELS = {
    "server": 123456789,
    "moderation": 123456789,
    "members": 123456789,
    "channels": 123456789,
    "messages": 123456789,
    "bot": 123456789
}



# ---------------------------------------------------------------------------
# Load the LLM (default: model.pt)
# ---------------------------------------------------------------------------
llm = App()
llm.load(MODEL_PATH)
if llm.model is None or llm.tok is None:
    print(f"Could not load '{MODEL_PATH}'. Train and save a model with sylvie_llm.py first.")
    sys.exit(1)

if llm.stage != "sft":
    print("Warning: this model has not been instruction-tuned (stage 2); replies may be poor.")

llm_lock = asyncio.Lock()  # one generation at a time
histories = {}             # user_id -> list of {"role", "content"} messages


def generate_reply(history, message):
    """Blocking generation; run it in a thread. `history` is a list of role/content messages."""
    S, tok, model = llm.S, llm.tok, llm.model
    msgs = with_system(history + [{"role": "user", "content": message}], S["system_prompt"])
    # leave room for the reply inside the context window; oldest turns are dropped first
    budget = model.c["block_size"] - min(int(S["max_new_tokens"]), model.c["block_size"] // 2)
    ids = chat_prompt_ids(tok, msgs, budget)

    out = []
    for t in model.generate(ids, int(S["max_new_tokens"]), float(S["temperature"]),
                            int(S["top_k"]), float(S["top_p"]), float(S["rep_penalty"]),
                            stop_ids=(tok.end_id, tok.eos_id), ban_ids=tuple(tok.role_ids.values())):
        if t in (tok.end_id, tok.eos_id):
            break
        out.append(t)
    return tok.decode(out).strip()


async def chat_with_model(user_id, message):
    history = histories.setdefault(user_id, [])
    async with llm_lock:
        reply = await asyncio.to_thread(generate_reply, list(history), message)
    if reply:  # don't store empty replies; they would poison later turns
        history += [{"role": "user", "content": message}, {"role": "assistant", "content": reply}]
        del history[:-2 * MAX_TURNS]
    print(reply)
    return (reply or "(no response)")[:1900]  # Discord limit is 2000 chars


# ---------------------------------------------------------------------------
# Discord bot
# ---------------------------------------------------------------------------
intents = discord.Intents.default()
intents.message_content = True


NO_PINGS = discord.AllowedMentions.none()  # model output can't ping @everyone/users


class SylvieBot(commands.Bot):
    async def setup_hook(self):
        await self.tree.sync()  # register slash commands


bot = SylvieBot(command_prefix="!", intents=intents)


# ---- slash commands --------------------------------------------------------
@bot.tree.command(name="chat", description="Chat with the AI model")
@app_commands.describe(message="What you want to say")
async def chat_slash(interaction: discord.Interaction, message: str):
    await interaction.response.defer()  # generation can take a while
    try:
        reply = await chat_with_model(interaction.user.id, message)
    except Exception as e:
        await interaction.followup.send(f"Error: `{type(e).__name__}: {e}`")
        return
    await interaction.followup.send(reply, allowed_mentions=NO_PINGS)

@bot.tree.command(name="resetchat", description="Clear your chat history with the AI")
async def resetchat_slash(interaction: discord.Interaction):
    histories.pop(interaction.user.id, None)
    await interaction.response.send_message("Chat history cleared.", ephemeral=True)

# ---- prefix commands -------------------------------------------------------
@bot.command(name="chat")
async def chat_prefix(ctx, *, message: str):
    async with ctx.typing():
        try:
            reply = await chat_with_model(ctx.author.id, message)
        except Exception as e:
            await ctx.reply(f"Error: `{type(e).__name__}: {e}`")
            return
    await ctx.reply(reply, allowed_mentions=NO_PINGS)

@bot.command(name="resetchat")
async def resetchat_prefix(ctx):
    histories.pop(ctx.author.id, None)
    await ctx.reply("Chat history cleared.")

# ---- mention the bot to chat -----------------------------------------------
@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return
    await bot.process_commands(message)
    if bot.user in message.mentions and not message.content.startswith(bot.command_prefix):
        text = message.content.replace(f"<@{bot.user.id}>", "").replace(f"<@!{bot.user.id}>", "").strip()
        if not text:
            return
        async with message.channel.typing():
            try:
                reply = await chat_with_model(message.author.id, text)
            except Exception as e:
                await message.reply(f"Error: `{type(e).__name__}: {e}`")
                return
        await message.reply(reply, allowed_mentions=NO_PINGS)

@bot.event
async def on_ready():
    print(f"Bot online: {bot.user}")

@bot.command()
async def test(ctx):
    await ctx.send("Hello!")

@bot.command()
async def repeat(ctx, *, message: str):
    await ctx.send(message)
    
@bot.tree.command(name="userinfo")
async def userinfo(
    interaction: discord.Interaction,
    member: discord.Member
):
    await interaction.response.send_message(
        f"Username: {member.name}\n"
        f"ID: {member.id}\n"
        f"Nickname: {member.nick}"
    )

@bot.tree.command(name="avatar")
async def avatar(
    interaction: discord.Interaction,
    user: discord.User
):
    await interaction.response.send_message(
        user.display_avatar.url
    )

@bot.tree.command(name="serverinfo")
async def serverinfo(interaction: discord.Interaction):

    guild = interaction.guild

    await interaction.response.send_message(
        f"Server: {guild.name}\n"
        f"Members: {guild.member_count}\n"
        f"Server ID: {guild.id}"
    )

@bot.tree.command(name="announce")
@app_commands.checks.has_permissions(manage_messages=True)
async def announce(
    interaction: discord.Interaction,
    channel: discord.TextChannel,
    message: str
):

    await channel.send(message)

    await interaction.response.send_message(
        "Announcement sent!",
        ephemeral=True
    )

@bot.tree.command(name="addrole")
@app_commands.checks.has_permissions(manage_roles=True)
async def addrole(
    interaction: discord.Interaction,
    member: discord.Member,
    role: discord.Role
):
    # nobody may hand out a role at or above their own highest role (unless they own the server)
    if interaction.user.id != interaction.guild.owner_id and role >= interaction.user.top_role:
        await interaction.response.send_message("You can't assign a role higher than or equal to your own.",
                                                ephemeral=True)
        return

    await member.add_roles(role)

    await interaction.response.send_message(
        f"Added {role.name} to {member.mention}!"
    )

@bot.tree.command(name="ban")
@app_commands.checks.has_permissions(ban_members=True)
async def ban(
    interaction: discord.Interaction,
    member: discord.Member,
    reason: str = "No reason provided"
):

    await member.ban(reason=reason)

    await interaction.response.send_message(
        f"{member} has been banned.\nReason: {reason}"
    )

@bot.tree.command(name="unban")
@app_commands.checks.has_permissions(ban_members=True)
async def unban(
    interaction: discord.Interaction,
    user_id: str,
    reason: str = "No reason provided"
):
    user = await bot.fetch_user(int(user_id))

    await interaction.guild.unban(user, reason=reason)

    await interaction.response.send_message(
        f"{user} has been unbanned.\nReason: {reason}"
    )
    
@bot.tree.command(name="kick")
@app_commands.checks.has_permissions(kick_members=True)
async def kick(
    interaction: discord.Interaction,
    member: discord.Member,
    reason: str = "No reason provided"
):

    await member.kick(reason=reason)

    await interaction.response.send_message(
        f"{member} has been kicked."
    )

@bot.tree.command(name="unkick")
@app_commands.checks.has_permissions(kick_members=True)
async def unkick(
    interaction: discord.Interaction,
    member_id: str
):
    invite = await interaction.channel.create_invite(
        max_age=86400,
        max_uses=1
    )

    await interaction.response.send_message(
        f"Reinvitation link for `{member_id}`:\n{invite}",
        ephemeral=True
    )
    
from datetime import timedelta

@bot.tree.command(name="timeout")
@app_commands.checks.has_permissions(moderate_members=True)
async def timeout(
    interaction: discord.Interaction,
    member: discord.Member,
    minutes: int
):

    await member.timeout(
        timedelta(minutes=minutes),
        reason="Moderator action"
    )

    await interaction.response.send_message(
        f"{member.mention} has been timed out."
    )

@bot.tree.command(name="untimeout")
@app_commands.checks.has_permissions(moderate_members=True)
async def untimeout(
    interaction: discord.Interaction,
    member: discord.Member,
    reason: str = "No reason provided"
):

    await member.timeout(
        None,
        reason=reason
    )

    await interaction.response.send_message(
        f"{member.mention}'s timeout has been removed.\nReason: {reason}"
    )
    
@bot.tree.command(name="clear")
@app_commands.checks.has_permissions(manage_messages=True)
async def clear(
    interaction: discord.Interaction,
    amount: int
):

    channel = interaction.channel

    await interaction.response.defer(ephemeral=True)

    deleted = await channel.purge(limit=amount)

    await interaction.followup.send(
        f"Deleted {len(deleted)} messages."
    )

@bot.command()
@commands.has_permissions(manage_nicknames=True)
async def nickname(ctx, member: discord.Member, *, nickname: str):
    try:
        await member.edit(nick=nickname)
        await ctx.send(
            f"Successfully changed {member.mention}'s nickname to **{nickname}**!"
        )
    except discord.Forbidden:
        await ctx.send("I don't have permission to change that member's nickname.")
    except discord.HTTPException as e:
        await ctx.send(f"An error occurred: {e}")

@bot.command()
@commands.has_permissions(manage_nicknames=True)
async def resetnick(ctx, member: discord.Member):
    try:
        await member.edit(nick=None)
        await ctx.send(f"Reset {member.mention}'s nickname!")

    except discord.Forbidden:
        await ctx.send("I don't have permission to do that.")


async def send_log(bot, log_type, message):
    channel_id = LOG_CHANNELS.get(log_type)

    if channel_id is None:
        return

    channel = bot.get_channel(channel_id)

    if channel:
        await channel.send(message, allowed_mentions=NO_PINGS)  # logged user text must not ping anyone

@bot.event
async def on_message_edit(before, after):
    if before.author.bot:
        return

    if before.content == after.content:
        return

    await send_log(
        bot,
        "messages",
        f"**Message Edited**\n"
        f"User: {before.author}\n"
        f"Channel: {before.channel.mention}\n"
        f"Before: {before.content}\n"
        f"After: {after.content}"
    )

@bot.event
async def on_message_delete(message):
    if message.author.bot:
        return

    await send_log(
        bot,
        "messages",
        f"**Message Deleted**\n"
        f"User: {message.author}\n"
        f"Channel: {message.channel.mention}\n"
        f"Content: {message.content}"
    )
    
@bot.event
async def on_member_update(before, after):
    if before.nick != after.nick:
        await send_log(
            bot,
            "members",
            f"**Nickname Changed**\n"
            f"User: {after}\n"
            f"Before: {before.nick}\n"
            f"After: {after.nick}"
        )

@bot.event
async def on_member_join(member):
    await send_log(
        bot,
        "members",
        f"**Member Joined**\n"
        f"User: {member}\n"
        f"ID: {member.id}"
    )
    
@bot.event
async def on_member_remove(member):
    await send_log(
        bot,
        "members",
        f"**Member Left**\n"
        f"User: {member}\n"
        f"ID: {member.id}"
    )

@bot.tree.error
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    if isinstance(error, app_commands.MissingPermissions):
        msg = "You don't have permission to use that command."
    else:
        msg = f"Error: `{type(error).__name__}: {error}`"
    if interaction.response.is_done():
        await interaction.followup.send(msg, ephemeral=True)
    else:
        await interaction.response.send_message(msg, ephemeral=True)

@bot.tree.interaction_check
async def interaction_check(interaction: discord.Interaction):

    await send_log(
        bot,
        "bot",
        f"**Bot Interaction**\n"
        f"User: {interaction.user}\n"
        f"User ID: {interaction.user.id}\n"
        f"Guild: {interaction.guild}\n"
        f"Command: {interaction.command.name if interaction.command else 'Unknown'}"
    )

    return True

@bot.event
async def on_command(ctx):
    await send_log(
        bot,
        "bot",
        f"**Prefix Command Used**\n"
        f"User: {ctx.author}\n"
        f"Command: {ctx.command}\n"
        f"Arguments: {ctx.message.content}"
    )

import time
import datetime

@bot.command()
async def uptime(ctx):
    # Calculate uptime
    current_time = time.time()
    uptime_seconds = int(round(current_time - start_time))
    uptime_formatted = str(datetime.timedelta(seconds=uptime_seconds))
    
    await ctx.send(f"Uptime: **{uptime_formatted}**")

@bot.command()
async def ping(ctx):
    ping_ms = round(bot.latency * 1000)
    await ctx.send(f"Ping: **{ping_ms} ms**")

if __name__ == "__main__":
    token = os.environ.get("DISCORD_TOKEN")
    if not token:
        token = input("Input your bot's token: ").strip()
        
    # Record start time when the script runs
    start_time = time.time()
    bot.run(token)

