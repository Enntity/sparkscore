# What I want from my reading

This is a description of one reader of Hacker News: what they open, what they skip, and why. Use it to decide whether a post is worth this reader's time.

## Who I am

I have written software for twenty years, mostly backend systems and databases, and I now run a small team. I read Hacker News in the morning for about fifteen minutes. I do not want to be informed about everything. I want three or four things a day that teach me something I can use or that are a pleasure to read closely.

## What I open every time

**Databases and storage.** Anything about how Postgres, SQLite, MySQL, DuckDB or a key-value store works inside: query planners, indexes, write-ahead logs, vacuum, replication, page formats. Benchmarks are welcome when the author shows the method and the numbers. War stories about migrations, corrupted data and recovery are the best posts on the site.

**Performance work with measurements.** Someone made a thing faster or smaller and shows before and after: milliseconds, megabytes, flame graphs. I do not care about the language. I care that something was measured.

**Reverse engineering and hardware hacking.** Dumping firmware, glitching a chip, decoding an undocumented protocol, repairing or modifying a device the maker wanted sealed. Old consoles, car computers, printers, lab equipment, all of it.

**Compilers, interpreters and operating systems.** Parsers, garbage collectors, linkers, schedulers, file systems, kernels. Toy implementations are fine if the author explains what they learned.

**Postmortems.** A real outage or security incident, written up afterwards by the people who were there, with a timeline and a cause. A status page that only says a service is down is not a postmortem and I skip it.

**Small tools somebody built for themselves.** A single binary, a self-hosted service, a command line utility, a library with a clear purpose. I like "Show HN" posts when one person made the thing and can say what it does in a sentence.

**Retrocomputing and the history of computing.** Old machines, old software, oral histories, how a famous system was really built.

**Mathematics and algorithms explained well.** A data structure, a proof, a clever encoding, written so a programmer can follow.

## What I skip every time

**Artificial intelligence as news.** Model launches, funding rounds, benchmark leaderboards, executives predicting the future, essays about whether AI will take jobs, agent frameworks and prompt tricks. I use these tools at work and I am tired of reading about them. The one exception is a post that explains how something works inside a model or an inference engine, with detail; that I will open.

**Company and money news.** Acquisitions, layoffs, earnings, lawsuits between large firms, stock prices, a chief executive being hired or fired, anything whose source is a business newspaper.

**Politics and world news.** Elections, governments, wars, court rulings, regulation, unless the post is a technical analysis of a specific system.

**Opinion pieces without evidence.** "X is dead", "X considered harmful", "why you should never do Y", anything that is a mood and not a finding. Rants about a company being awful.

**Productivity and self-improvement.** Morning routines, note-taking systems, career advice, how to be a better manager, burnout essays.

**Cryptocurrency, in every form.**

**Health and lifestyle science.** Diet, sleep, exercise and psychology studies reported from a press release.

**Product announcements from large companies.** A new phone, a new laptop, a new cloud service, a version number, a pricing change.

## How to weigh a post I am unsure about

If a post falls between these lists, ask whether it contains something concrete that a working programmer could learn from: a number, a diagram, a piece of code, a cause. If it does, I probably want it. If it is mainly an opinion, an announcement or a reaction to the news, I probably do not.

A dry title is a good sign. A title that tells me how to feel is a bad sign.

Security posts are welcome when they describe a specific vulnerability or attack and how it works. They are not welcome when they are a general warning or a vendor's marketing.

Posts about programming languages are welcome when they are about implementation or a measured comparison, and not welcome when they are arguments about which language is best.

When in doubt, leave it out. I would rather miss a good post than have my fifteen minutes filled with average ones.
