# VPS Setup Runbook

A step-by-step guide for setting up and securing a new OVH Ubuntu VPS with CI/CD deployment.

---

## Prerequisites

Before starting, make sure you have:
- SSH access to the VPS (root or default user)
- A local machine with WSL or Linux terminal
- Access to your GitHub repository
- Access to the OVH Control Panel

---

## 1. Generate a local SSH key (on your local machine)

```bash
ssh-keygen -t ed25519 -C "your@email.com"
# Accept default location (~/.ssh/id_ed25519)
# Set a strong passphrase
```

---

## 2. Create a new sudo user

```bash
# Replace 'deploy-user' with your chosen username
NEW_USER="deploy-user"

adduser $NEW_USER
usermod -aG sudo $NEW_USER

# Copy SSH key from root to new user
mkdir -p /home/$NEW_USER/.ssh
cp ~/.ssh/authorized_keys /home/$NEW_USER/.ssh/
chown -R $NEW_USER:$NEW_USER /home/$NEW_USER/.ssh
chmod 700 /home/$NEW_USER/.ssh
chmod 600 /home/$NEW_USER/.ssh/authorized_keys
```

---

## 3. SSH hardening

```bash
SSH_PORT=2277  # Pick any port between 1024-65535

# Back up original config
cp /etc/ssh/sshd_config /etc/ssh/sshd_config.bak

# Disable ssh.socket (Ubuntu 22.04+ overrides sshd_config otherwise)
systemctl disable --now ssh.socket
systemctl enable --now ssh

# Apply hardened settings
cat >> /etc/ssh/sshd_config << EOF

# --- Hardening ---
Port $SSH_PORT
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
AllowUsers $NEW_USER
MaxAuthTries 3
LoginGraceTime 30
X11Forwarding no
EOF

# Verify config
sshd -t

# Restart SSH
systemctl restart sshd

# Confirm new port is active
ss -tlnp | grep sshd
```

> ⚠️ **Before continuing:** Open a new terminal and confirm login works on the new port:
> `ssh -p 2277 deploy-user@YOUR_SERVER_IP`
> Only proceed once you are in.

---

## 4. Firewall (ufw)

```bash
SSH_PORT=2277  # Must match the port set above

ufw default deny incoming
ufw default allow outgoing
ufw allow $SSH_PORT/tcp

# Add any other ports your app needs, e.g.:
# ufw allow 80/tcp
# ufw allow 443/tcp

ufw enable
ufw status verbose
```

---

## 5. fail2ban

```bash
apt install fail2ban -y

cat > /etc/fail2ban/jail.local << EOF
[DEFAULT]
bantime  = 1h
findtime = 10m
maxretry = 3
backend  = systemd

[sshd]
enabled  = true
port     = $SSH_PORT
logpath  = %(sshd_log)s
maxretry = 3
bantime  = 24h
EOF

systemctl enable fail2ban
systemctl restart fail2ban

# Verify
fail2ban-client status sshd
```

---

## 6. Automatic security updates

```bash
apt install unattended-upgrades apt-listchanges -y

cat > /etc/apt/apt.conf.d/20auto-upgrades << EOF
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
APT::Periodic::AutocleanInterval "7";
EOF

dpkg-reconfigure -plow unattended-upgrades
systemctl status unattended-upgrades
```

---

## 7. Add deploy-user to docker group

```bash
sudo usermod -aG docker deploy-user
# Log out and back in for this to take effect
```

---

## 8. OVH Control Panel

1. Log into the **OVH Control Panel**
2. Go to your VPS → **Firewall Network**
3. Add a rule to **allow TCP port 2277** inbound
4. Remove or block port 22

---

## 9. Set up deployment keys on the server

Log in as deploy-user:

```bash
ssh -p 2277 deploy-user@YOUR_SERVER_IP
```

Generate a dedicated GitHub deployment key:

```bash
ssh-keygen -t ed25519 -C "github-deploy" -f ~/.ssh/id_ed25519 -N ""

# Add public key to authorized_keys
cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys

# Add GitHub to known hosts
ssh-keyscan github.com >> ~/.ssh/known_hosts
chmod 600 ~/.ssh/known_hosts

# Test GitHub connection
ssh -T git@github.com
```

Add the public key to GitHub:
- Go to your repo → **Settings** → **Deploy keys** → **Add deploy key**
- Paste the contents of `~/.ssh/id_ed25519.pub`
- Read-only unless your deploy script needs to push

---

## 10. Clone the repositories

```bash
# Production
git clone git@github.com:YOUR_USERNAME/YOUR_REPO.git ~/YOUR_REPO

# Acceptance
git clone git@github.com:YOUR_USERNAME/YOUR_REPO.git ~/YOUR_REPO-acceptance
cd ~/YOUR_REPO-acceptance
git checkout develop
```

---

## 11. Set GitHub secrets

Go to your repo → **Settings** → **Secrets and variables** → **Actions** and add:

### Production secrets

| Secret | Value |
|---|---|
| `VPS_HOST` | Your server IP |
| `VPS_USER` | `deploy-user` |
| `VPS_PORT` | `2277` |
| `VPS_SSH_KEY` | Contents of `~/.ssh/id_ed25519` (private key, full including header/footer) |
| `VPS_KNOWN_HOSTS` | Output of `ssh-keyscan -p 2277 YOUR_SERVER_IP` run from local machine |
| `VPS_APP_DIR` | `/home/deploy-user/YOUR_REPO` |
| `VPS_COMPOSE_FILE` | e.g. `docker-compose.prod.yml` |
| `VPS_BRANCH` | `main` |

### Acceptance secrets

| Secret | Value |
|---|---|
| `ACCEPTANCE_VPS_APP_DIR` | `/home/deploy-user/YOUR_REPO-acceptance` |
| `ACCEPTANCE_VPS_BRANCH` | `develop` |
| `ACCEPTANCE_VPS_COMPOSE_FILE` | e.g. `docker-compose.acceptance.yml` |

> The acceptance pipeline reuses `VPS_HOST`, `VPS_USER`, `VPS_PORT`, `VPS_SSH_KEY`, and `VPS_KNOWN_HOSTS` from the production secrets.

---

## 12. Test the pipelines

**Production** — push to main:
```bash
git commit --allow-empty -m "test: trigger deployment pipeline"
git push origin main
```

**Acceptance** — push to develop:
```bash
git checkout develop
git commit --allow-empty -m "test: trigger acceptance deployment"
git push origin develop
```

Watch the **Actions** tab on GitHub for results.

---

## 13. Final verification

```bash
# Firewall is up
ufw status verbose

# fail2ban is running
fail2ban-client status sshd

# SSH is on the correct port
ss -tlnp | grep sshd

# Root login is disabled
grep "PermitRootLogin" /etc/ssh/sshd_config

# No unexpected users
cat /etc/passwd | grep -v nologin | grep -v false

# Auto-updates are active
systemctl status unattended-upgrades
```

---

## Summary

| Setting | Value |
|---|---|
| SSH port | 2277 |
| Root login | Disabled |
| Password login | Disabled |
| Allowed SSH user | `deploy-user` |
| fail2ban ban threshold | 3 failed attempts |
| fail2ban ban duration | 24 hours |
| Auto security updates | Daily |
| Production branch | `main` |
| Acceptance branch | `develop` |
