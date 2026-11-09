// The Android launcher (dm/android): installs a game's RomFS image from the parts scripts/split_romfs.py writes, and will start the host once it runs on Android.
pluginManagement {
    repositories { google(); mavenCentral(); gradlePluginPortal() }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories { google(); mavenCentral() }
}
rootProject.name = "NxNativeLauncher"
include(":app")
