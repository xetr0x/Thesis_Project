# Add project specific ProGuard rules here.
# You can control the set of applied configuration files using the
# proguardFiles setting in build.gradle.
#
# For more details, see
#   http://developer.android.com/guide/developing/tools/proguard.html

# If your project uses WebView with JS, uncomment the following
# and specify the fully qualified class name to the JavaScript interface
# class:
#-keepclassmembers class fqcn.of.javascript.interface.for.webview {
#   public *;
#}
# In the [versions] section, specify the versions of the libraries
[versions]
paho = "1.2.5"
androidx-core = "1.8.0"
androidx-appcompat = "1.4.0"
material = "1.4.0"

# In the [libraries] section, reference the libraries by their module names
[libraries]
androidx-core-ktx = { module = "androidx.core:core-ktx", version.ref = "androidx-core" }
androidx-appcompat = { module = "androidx.appcompat:appcompat", version.ref = "androidx-appcompat" }
material = { module = "com.google.android.material:material", version.ref = "material" }
paho-mqtt-client = { module = "org.eclipse.paho:org.eclipse.paho.client.mqttv3", version.ref = "paho" }
paho-android-service = { module = "org.eclipse.paho:android-service", version.ref = "paho" }
# Uncomment this to preserve the line number information for
# debugging stack traces.
#-keepattributes SourceFile,LineNumberTable

# If you keep the line number information, uncomment this to
# hide the original source file name.
#-renamesourcefileattribute SourceFile