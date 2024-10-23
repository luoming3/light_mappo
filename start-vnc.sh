#!/bin/sh

# start vnc server
vncserver -kill :*
vncserver :77 -localhost no

# start novnc server
ps -ef | grep websockify | awk '{print $2}' | xargs kill -9 || echo ""
websockify --web /usr/share/novnc 8077 localhost:5977