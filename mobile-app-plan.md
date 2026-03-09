# Mobile App Plan: Eenkaartjeleggen - Klaverjassen

**Goal:** Publish the existing web application to the Google Play Store with minimal changes to the existing codebase. iOS / App Store support is kept as a future option.

**Approach:** Capacitor WebView wrapper + PWA manifest
**App name:** Eenkaartjeleggen - Klaverjassen
**Backend:** Live server at `https://eenkaartjeleggen.nl` (no backend changes)
**Repo layout:** Capacitor project lives in `mobile/` inside this repo

---

## Table of Contents

1. [Overview](#1-overview)
2. [Prerequisites & Accounts](#2-prerequisites--accounts)
3. [Phase 1 — PWA Manifest (web app changes)](#3-phase-1--pwa-manifest-web-app-changes)
4. [Phase 2 — Capacitor Project Setup](#4-phase-2--capacitor-project-setup)
5. [Phase 3 — App Icons & Store Assets](#5-phase-3--app-icons--store-assets)
6. [Phase 4 — Android (Google Play)](#6-phase-4--android-google-play)
7. [Phase 5 — iOS (Apple App Store) — Future](#7-phase-5--ios-apple-app-store--future)
8. [Phase 6 — CI/CD Integration](#8-phase-6--cicd-integration)
9. [Store Listing Content](#9-store-listing-content)
10. [Ongoing Maintenance](#10-ongoing-maintenance)

---

## 1. Overview

The app is a thin native shell that loads `https://eenkaartjeleggen.nl` in a full-screen WebView. No game logic moves to the client. The web app itself receives a PWA manifest so it is also installable directly from the browser.

```
┌──────────────────────────────────┐
│  Native Shell (Capacitor)         │  Android .aab / iOS .ipa
│  ┌────────────────────────────┐  │
│  │  WebView                   │  │
│  │  https://eenkaartjeleggen.nl│  │
│  └────────────────────────────┘  │
└──────────────────────────────────┘
```

**What changes in the existing repo:**

| File / location | Change |
|-----------------|--------|
| `templates/index.html` | Add `<link rel="manifest">` and theme-color meta tag |
| `static/manifest.webmanifest` | New file (PWA manifest) |
| `static/icons/` | New folder with generated icon set |
| `mobile/` | New folder — entire Capacitor project |
| `.github/workflows/` | Optional: add mobile build jobs |

---

## 2. Prerequisites & Accounts

### 2.1 Google Play (you have this)

- Confirm your Google Play Developer account is active at [play.google.com/console](https://play.google.com/console).
- You will need to create a new app entry (section 6).

### 2.2 Apple Developer (skipped for now — future option)

iOS support is intentionally deferred. When you are ready to add it:

1. Go to [developer.apple.com/programs/enroll](https://developer.apple.com/programs/enroll/).
2. Sign in with your Apple ID (create one if needed).
3. Enroll as an **Individual** (unless publishing under a company name — requires D-U-N-S number).
4. Pay the **$99/year** fee.
5. Wait for approval — typically 24–48 hours, occasionally up to a week.
6. Then follow [Phase 5](#7-phase-5--ios-apple-app-store--future).

> Because the Capacitor project already includes `npx cap add ios` in Phase 2, the native iOS project will be generated and ready. Adding iOS later requires no rework of the Android or web layers.

### 2.3 Local toolchain

| Tool | Required for | Install |
|------|-------------|---------|
| Node.js 18+ | Capacitor CLI | [nodejs.org](https://nodejs.org) |
| npm 9+ | Capacitor CLI | bundled with Node |
| Android Studio (latest) | Android build & signing | [developer.android.com/studio](https://developer.android.com/studio) |
| Java 17 (JDK) | Android build | bundled with Android Studio |

> Xcode (macOS only) is only needed when you add iOS later — skip it for now.

---

## 3. Phase 1 — PWA Manifest (web app changes)

These are the only changes to the existing Flask application.

### 3.1 Create `static/manifest.webmanifest`

```json
{
  "name": "Eenkaartjeleggen - Klaverjassen",
  "short_name": "Klaverjassen",
  "description": "Multiplayer Klaverjassen kaartspel. Speel online met vrienden of tegen de computer.",
  "start_url": "/",
  "display": "standalone",
  "orientation": "portrait",
  "background_color": "#1a5c2a",
  "theme_color": "#1a5c2a",
  "lang": "nl",
  "icons": [
    { "src": "/static/icons/icon-192.png", "sizes": "192x192", "type": "image/png" },
    { "src": "/static/icons/icon-512.png", "sizes": "512x512", "type": "image/png" },
    { "src": "/static/icons/icon-maskable-512.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable" }
  ]
}
```

> Adjust `background_color` and `theme_color` to match the game's green table color. The values above are a placeholder — confirm the hex code from `static/style.css`.

### 3.2 Update `templates/index.html`

Add the following inside `<head>`:

```html
<link rel="manifest" href="/static/manifest.webmanifest">
<meta name="theme-color" content="#1a5c2a">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Klaverjassen">
<link rel="apple-touch-icon" href="/static/icons/icon-180.png">
```

### 3.3 Serve the manifest from Flask

Flask already serves `/static/` files. No route changes are needed. Verify the manifest is reachable at `https://eenkaartjeleggen.nl/static/manifest.webmanifest` after deploy.

---

## 4. Phase 2 — Capacitor Project Setup

The Capacitor project lives in `mobile/` and points to the live server. It does **not** bundle the web app — it only wraps it.

### 4.1 Initialise the project

```bash
mkdir mobile && cd mobile
npm init -y
npm install @capacitor/core @capacitor/cli @capacitor/android @capacitor/ios
npx cap init "Eenkaartjeleggen - Klaverjassen" "nl.eenkaartjeleggen.app" --web-dir www
```

### 4.2 Create `mobile/www/index.html` (redirect shell)

Since the app points to the live server, the local `www/` directory only needs a redirect:

```html
<!DOCTYPE html>
<html>
<head>
  <meta http-equiv="refresh" content="0; url=https://eenkaartjeleggen.nl">
</head>
<body></body>
</html>
```

### 4.3 Configure `mobile/capacitor.config.ts`

```typescript
import { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'nl.eenkaartjeleggen.app',
  appName: 'Eenkaartjeleggen - Klaverjassen',
  webDir: 'www',
  server: {
    url: 'https://eenkaartjeleggen.nl',
    cleartext: false,
  },
  android: {
    allowMixedContent: false,
  },
  ios: {
    contentInset: 'automatic',
  },
};

export default config;
```

> The `server.url` field tells Capacitor to load the live URL instead of a local bundle. This is the key setting that keeps the mobile app in sync with the web server without resubmitting to the stores on every update.

### 4.4 Add native platforms

```bash
cd mobile
npx cap add android
# npx cap add ios     # Skipped for now — run this when ready for iOS (Phase 5)
npx cap sync
```

### 4.5 Add `mobile/` to `.gitignore` exceptions

The generated `android/` and `ios/` folders inside `mobile/` contain build artifacts. Add to the root `.gitignore`:

```
# Keep Capacitor native projects
!mobile/android/
!mobile/ios/
mobile/node_modules/
mobile/www/
```

---

## 5. Phase 3 — App Icons & Store Assets

### 5.1 Source image requirements

Create one **master icon** at **1024×1024 px**, PNG format, no transparency (use a solid background). Suggested design: a playing card or suit symbol on the game's green background.

### 5.2 Generate all required sizes

Use the [@capacitor/assets](https://github.com/ionic-team/capacitor-assets) tool to generate all platform icon sizes from the master:

```bash
cd mobile
npm install -D @capacitor/assets
# Place your 1024x1024 source icon at:
#   mobile/resources/icon.png
#   mobile/resources/icon-foreground.png  (for Android adaptive icon foreground)
#   mobile/resources/icon-background.png  (for Android adaptive icon background)
npx capacitor-assets generate
```

This generates all required sizes for both Android and iOS automatically.

### 5.3 Required sizes reference (generated automatically)

| Platform | Size | Purpose |
|----------|------|---------|
| iOS | 1024×1024 | App Store listing |
| iOS | 180×180 | iPhone home screen |
| iOS | 167×167 | iPad Pro |
| Android | 512×512 | Play Store listing |
| Android | 192×192 | Launcher (xxxhdpi) |
| Android | Adaptive | Foreground + background layers |
| PWA | 192×192 | Browser install prompt |
| PWA | 512×512 | Splash / maskable |

### 5.4 Store screenshots

Both stores require screenshots. Capture them from the live web app at the required resolutions:

**Google Play (required):**
- At least 2 phone screenshots: **1080×1920 px** (portrait)
- Optional: 7-inch tablet, 10-inch tablet

**Apple App Store (required):**
- iPhone 6.5" display: **1284×2778 px** (portrait) — 3 minimum
- iPhone 5.5" display: **1242×2208 px** — 3 minimum
- iPad Pro 12.9": **2048×2732 px** — if you want iPad support

> Use browser dev tools to simulate these resolutions against `eenkaartjeleggen.nl`, then screenshot. Tools like [screely.com](https://screely.com) can add device frames for store presentation.

### 5.5 Feature graphic (Google Play only)

Google Play requires a **1024×500 px** feature graphic (banner image shown on the store listing). Design one with the app name and a card game visual.

---

## 6. Phase 4 — Android (Google Play)

### 6.1 Open the Android project

```bash
cd mobile
npx cap open android
```

Android Studio opens the project at `mobile/android/`.

### 6.2 Configure app metadata

In `mobile/android/app/build.gradle`, verify:

```groovy
android {
    defaultConfig {
        applicationId "nl.eenkaartjeleggen.app"
        minSdk 26          // Android 8.0 — covers ~95% of active devices
        targetSdk 34       // Must match current Play Store requirement
        versionCode 1
        versionName "1.0.0"
    }
}
```

### 6.3 Create a release signing keystore

Run once; store the keystore file and passwords securely (password manager, not in git):

```bash
keytool -genkey -v \
  -keystore eenkaartjeleggen-release.jks \
  -alias eenkaartjeleggen \
  -keyalg RSA -keysize 2048 \
  -validity 10000
```

Add to `mobile/android/app/build.gradle`:

```groovy
android {
    signingConfigs {
        release {
            storeFile file("../../eenkaartjeleggen-release.jks")
            storePassword System.getenv("KEYSTORE_PASSWORD")
            keyAlias "eenkaartjeleggen"
            keyPassword System.getenv("KEY_PASSWORD")
        }
    }
    buildTypes {
        release {
            signingConfig signingConfigs.release
            minifyEnabled false
        }
    }
}
```

> **Never commit the `.jks` file to git.** Add it to `.gitignore`.

### 6.4 Build the release bundle

```bash
cd mobile/android
./gradlew bundleRelease
# Output: mobile/android/app/build/outputs/bundle/release/app-release.aab
```

### 6.5 Create the Play Store listing

1. Go to [play.google.com/console](https://play.google.com/console) → **Create app**
2. App name: `Eenkaartjeleggen - Klaverjassen`
3. Default language: **Dutch (nl)**
4. App or game: **Game**
5. Free or paid: **Free**
6. Fill in the **Store listing** (see section 9 for copy)
7. Upload the `.aab` to **Internal testing** track first
8. Complete the **Content rating questionnaire** (select "Card game", no violence/gambling)
9. Complete the **Data safety** section (the app connects to your own server; no third-party data collection)
10. Promote from Internal → Production when ready

### 6.6 Google Play review timeline

- Internal testing: instant
- Production review: typically **3–7 business days** for first submission

---

## 7. Phase 5 — iOS (Apple App Store) — Future

> **Skipped for now.** Come back to this phase once you have registered an Apple Developer account (see section 2.2). No rework of the Android or web layers is required when you pick this up.

> **Requires a Mac.** If you do not have one, use a GitHub Actions macOS runner or a service like MacStadium for the build step.

### 7.1 Open the iOS project

```bash
cd mobile
npx cap open ios
```

Xcode opens `mobile/ios/App/App.xcworkspace`.

### 7.2 Configure signing in Xcode

1. Select the **App** target → **Signing & Capabilities**
2. Team: select your Apple Developer account
3. Bundle Identifier: `nl.eenkaartjeleggen.app`
4. Enable **Automatically manage signing**

### 7.3 Configure app metadata in Xcode

In `mobile/ios/App/App/Info.plist`, verify or set:
- `CFBundleDisplayName`: `Klaverjassen`
- `CFBundleVersion`: `1`
- `CFBundleShortVersionString`: `1.0.0`
- `NSAllowsArbitraryLoads`: `false` (HTTPS only — already enforced by your server)

### 7.4 Build & archive

1. In Xcode, set scheme to **App** and destination to **Any iOS Device**
2. **Product → Archive**
3. In the Organizer window, select the archive → **Distribute App**
4. Choose **App Store Connect** → **Upload**

### 7.5 Create the App Store listing

1. Go to [appstoreconnect.apple.com](https://appstoreconnect.apple.com) → **My Apps → +**
2. App name: `Eenkaartjeleggen - Klaverjassen`
3. Primary language: **Dutch**
4. Bundle ID: `nl.eenkaartjeleggen.app`
5. SKU: `eenkaartjeleggen-001`
6. Fill in **App Information** and **Pricing** (Free)
7. Upload screenshots (see section 5.4)
8. Fill in the **App Store description** (see section 9)
9. Submit for review

### 7.6 Apple App Review timeline

- First submission: **1–3 business days** (can be longer)
- Apple may request a demo account or ask clarifying questions about the game
- Common rejection reason for WebView apps: **"app does not provide enough functionality"** — mitigate by ensuring the full game is playable offline (or explicitly state it requires network)

> **Apple WebView policy note:** Apple guideline 4.2 discourages thin web wrappers. Your app is more likely to pass review because it wraps a fully featured, original multiplayer game — not a simple website. If rejected, the response is to appeal with a description of the game's unique functionality.

---

## 8. Phase 6 — CI/CD Integration

### 8.1 Android build in GitHub Actions

Add `.github/workflows/android-build.yml`:

```yaml
name: Android Build

on:
  push:
    branches: [main]
    paths:
      - 'mobile/**'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '18'
      - run: npm ci
        working-directory: mobile
      - run: npx cap sync android
        working-directory: mobile
      - uses: actions/setup-java@v4
        with:
          java-version: '17'
          distribution: 'temurin'
      - name: Build release AAB
        working-directory: mobile/android
        env:
          KEYSTORE_PASSWORD: ${{ secrets.KEYSTORE_PASSWORD }}
          KEY_PASSWORD: ${{ secrets.KEY_PASSWORD }}
        run: ./gradlew bundleRelease
      - uses: actions/upload-artifact@v4
        with:
          name: app-release.aab
          path: mobile/android/app/build/outputs/bundle/release/app-release.aab
```

**Required GitHub Secrets for Android:**

| Secret | Value |
|--------|-------|
| `KEYSTORE_PASSWORD` | Keystore password |
| `KEY_PASSWORD` | Key password |
| `KEYSTORE_FILE_BASE64` | Base64-encoded `.jks` file |

### 8.2 iOS build (macOS runner) — Future

Skipped for now. Add this job after completing Phase 5. iOS CI builds require a macOS runner and Xcode signing certificates stored as GitHub Secrets — document that setup at the time of iOS onboarding.

---

## 9. Store Listing Content

### App name
`Eenkaartjeleggen - Klaverjassen`

### Short description (Google Play, max 80 chars)
`Speel Klaverjassen online met vrienden of tegen slimme AI-tegenstanders.`

### Full description

```
Eenkaartjeleggen is een gratis multiplayer Klaverjassen kaartspel voor 2 tot 4 spelers.

Maak een kamer aan, deel de code met vrienden en speel direct vanuit je browser of deze app. Lege plaatsen worden automatisch gevuld door AI-tegenstanders op drie niveaus: beginner, gevorderd en expert.

KENMERKEN
• Multiplayer via gedeelde kamercodes
• AI-tegenstanders op drie moeilijkheidsniveaus
• Rotterdam- en Amsterdam-spelregels
• Meerdere spelmodi: puntenlimiet, boom of vrij spelen
• Volledige score- en roembijhouding
• Nederlandstalige en Engelstalige interface
• Herverbinden na verbroken verbinding

Klaverjassen is een traditioneel Nederlands kaartspel waarbij twee teams van twee spelers strijden om troef te verklaren en trucs te winnen.
```

### Category
- Google Play: **Card Games** (under Board)
- App Store: **Games → Card**

### Content rating
- Google Play: **Everyone**
- App Store: **4+**

### Keywords (App Store)
`klaverjassen, kaartspel, klaverjas, multiplayer, troef, nederland, card game`

---

## 10. Ongoing Maintenance

### Updating the app

Because the Capacitor wrapper loads the live server URL, **game updates deploy automatically** with the existing CI/CD pipeline — no store resubmission needed.

Store resubmission is only required when:
- The native shell itself changes (Capacitor version upgrade)
- New native permissions are added
- App metadata or screenshots need updating
- `versionCode` / `versionName` bumps are needed for store compliance

### Version naming convention

| Field | Purpose | Example |
|-------|---------|---------|
| `versionName` | Human-readable, shown in stores | `1.0.0` |
| `versionCode` (Android) / `CFBundleVersion` (iOS) | Integer, increments with every upload | `1`, `2`, `3` … |

### Checklist for future store updates

- [ ] Increment `versionCode` in `mobile/android/app/build.gradle`
- [ ] Increment `CFBundleVersion` in `mobile/ios/App/App/Info.plist`
- [ ] Run `npx cap sync` to pull any Capacitor plugin updates
- [ ] Build and test on a physical device before submitting
- [ ] Submit to internal/TestFlight track before promoting to production

---

*Plan written 2026-02-28. Apple App Store guidelines and Google Play policies may change — verify current requirements at submission time.*
