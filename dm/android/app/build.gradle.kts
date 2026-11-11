plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "dev.nxnative.fortunesweave"
    compileSdk = 35
    defaultConfig {
        applicationId = "dev.nxnative.fortunesweave"
        minSdk = 30
        targetSdk = 35
        versionCode = 1
        versionName = "0.1"
        // the launcher icon: game art kept out of the repository (art_res/ is git-ignored, filled on the build machine: mipmap-*/ic_launcher.png). Without it the
        // app takes Android's default icon, so a fresh checkout still builds
        manifestPlaceholders["appIcon"] = if (file("art_res/mipmap-mdpi/ic_launcher.png").isFile) "@mipmap/ic_launcher" else "@android:drawable/sym_def_app_icon"
    }
    sourceSets["main"].res.srcDirs("art_res")
    buildTypes {
        release { isMinifyEnabled = false }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    // the host program ships as a native library (libdmhost.so, built by dm/android/host/build.py into src/main/jniLibs): it must be unpacked to the native library
    // folder to be run, the one place Android lets an app execute a file it ships
    packaging { jniLibs { useLegacyPackaging = true } }
}

dependencies {
    testImplementation("junit:junit:4.13.2")
}
