package app.thesis_app

import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import android.widget.TextView
import androidx.core.view.WindowInsetsCompat
import com.hivemq.client.mqtt.MqttClient
import com.hivemq.client.mqtt.mqtt3.Mqtt3AsyncClient
import org.json.JSONObject

class MainActivity : AppCompatActivity() {
    private lateinit var mqttClient: Mqtt3AsyncClient
    private lateinit var time_placeholder: TextView
    private lateinit var temp_placeholder: TextView
    private lateinit var humidty_placeholder: TextView
    private lateinit var voc_placeholder: TextView
    private lateinit var ir_light_placeholder: TextView
    private lateinit var visable_light_placeholder: TextView
    private lateinit var co2: TextView



    private lateinit var status: TextView
    private lateinit var time: TextView
    private lateinit var temp: TextView
    private lateinit var humidty: TextView
    private lateinit var voc: TextView
    private lateinit var ir_light: TextView
    private lateinit var visable_light: TextView
    private lateinit var co2_placeholder: TextView



    private lateinit var time_placeholder2: TextView
    private lateinit var temp_placeholder2: TextView
    private lateinit var humidty_placeholder2: TextView
    private lateinit var voc_placeholder2: TextView
    private lateinit var ir_light_placeholder2: TextView
    private lateinit var visable_light_placeholder2: TextView
    private lateinit var co2_2: TextView



    private lateinit var status2: TextView
    private lateinit var time2: TextView
    private lateinit var temp2: TextView
    private lateinit var humidty2: TextView
    private lateinit var voc2: TextView
    private lateinit var ir_light2: TextView
    private lateinit var visable_light2: TextView
    private lateinit var co2_placeholder2: TextView


    private lateinit var resetButton: Button


    private val serverAdress = "213.65.96.164"
    private val port = 1883
    private val topic = "thesis/anomalies/1"
    private val topic2 = "thesis/anomalies/2"




    enum class AppState {
        ANOMALY_DETECTED_ROOM_ONE,
        ANOMALY_DETECTED_ROOM_TWO,
        NO_ANOMALY,
        NOT_CONNECTED
    }
    private var currentState = AppState.NOT_CONNECTED

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets

        }



        mqttClient = MqttClient.builder()
            .useMqttVersion3()
            .serverHost(serverAdress) // or your IP address
            .serverPort(port)
            .buildAsync()

        connectToBroker()

        //ROOM 1
        status = findViewById(R.id.status_text)

        time_placeholder = findViewById(R.id.time)
        temp_placeholder = findViewById(R.id.temp)
        humidty_placeholder = findViewById(R.id.humidity)
        voc_placeholder = findViewById(R.id.voc)
        ir_light_placeholder = findViewById(R.id.ir_light)
        visable_light_placeholder = findViewById(R.id.visable_light)
        co2_placeholder = findViewById(R.id.co2)


        time = findViewById(R.id.time_value)
        temp = findViewById(R.id.temp_value)
        humidty = findViewById(R.id.humidity_value)
        voc = findViewById(R.id.voc_value)
        ir_light = findViewById(R.id.ir_light_value)
        visable_light = findViewById(R.id.visable_light_value)
        co2 = findViewById(R.id.co2_value)
        resetButton = findViewById(R.id.reset_button)





      //ROOM 2
        status2 = findViewById(R.id.status_text2)

        time_placeholder2 = findViewById(R.id.time2)
        temp_placeholder2 = findViewById(R.id.temp2)
        humidty_placeholder2 = findViewById(R.id.humidity2)
        voc_placeholder2 = findViewById(R.id.voc2)
        ir_light_placeholder2 = findViewById(R.id.ir_light2)
        visable_light_placeholder2 = findViewById(R.id.visable_light2)
        co2_placeholder2 = findViewById(R.id.co2_2)


        time2 = findViewById(R.id.time_value2)
        temp2 = findViewById(R.id.temp_value2)
        humidty2 = findViewById(R.id.humidity_value2)
        voc2 = findViewById(R.id.voc_value2)
        ir_light2 = findViewById(R.id.ir_light_value2)
        visable_light2 = findViewById(R.id.visable_light_value2)
        co2_2 = findViewById(R.id.co2_value2)



        resetButton.setOnClickListener {
            switch_no_anomalies()
        }


    }

    private fun connectToBroker() {
        mqttClient.connect()
            .whenComplete { _, throwable ->
                if (throwable == null) {
                    runOnUiThread {

                        switch_no_anomalies()
                    }
                    subscribe_room_one(topic)
                    subscribe_room_two(topic2)
                }else {
                    AppState.NOT_CONNECTED
                    status.text = "Unable to connect"

                }
                }
    }


    private fun subscribe_room_one(topic: String) {
        mqttClient.subscribeWith()
            .topicFilter(topic)
            .callback { publish ->
                val data_string = String(publish.payloadAsBytes)
                Log.d("MQTT", "Received: $data_string")
                val data_json = JSONObject(data_string)
                Log.d("MQTT", "Received: $data_json")
                runOnUiThread { anomaly_detected_room_1(data_json) }


            }
            .send()
    }

    private fun subscribe_room_two(topic: String) {
        mqttClient.subscribeWith()
            .topicFilter(topic)
            .callback { publish ->
                val data_string = String(publish.payloadAsBytes)
                val data_json = JSONObject(data_string)
                Log.d("MQTT", "Received: $data_json")
                runOnUiThread { anomaly_detected_room_2(data_json) }


            }
            .send()
    }


    private fun switch_no_anomalies() {
        status.visibility = View.GONE
        time.visibility = View.GONE
        temp.visibility = View.GONE
        humidty.visibility = View.GONE
        voc.visibility = View.GONE
        ir_light.visibility = View.GONE
        visable_light.visibility = View.GONE
        co2.visibility = View.GONE

        time_placeholder.visibility = View.GONE
        temp_placeholder.visibility = View.GONE
        humidty_placeholder.visibility = View.GONE
        voc_placeholder.visibility = View.GONE
        ir_light_placeholder.visibility = View.GONE
        visable_light_placeholder.visibility = View.GONE
        co2_placeholder.visibility = View.GONE

        currentState = AppState.NO_ANOMALY
        status2.text = "No anomalies detected"
        time2.visibility = View.GONE
        temp2.visibility = View.GONE
        humidty2.visibility = View.GONE
        voc2.visibility = View.GONE
        ir_light2.visibility = View.GONE
        visable_light2.visibility = View.GONE
        co2_2.visibility = View.GONE

        time_placeholder2.visibility = View.GONE
        temp_placeholder2.visibility = View.GONE
        humidty_placeholder2.visibility = View.GONE
        voc_placeholder2.visibility = View.GONE
        ir_light_placeholder2.visibility = View.GONE
        visable_light_placeholder2.visibility = View.GONE
        co2_placeholder2.visibility = View.GONE

        resetButton.visibility = View.GONE

    }

    private fun anomaly_detected_room_1(data: JSONObject) {

        if (currentState!=AppState.ANOMALY_DETECTED_ROOM_TWO){
            status2.text = "No Anomalies Found In Room 2"
        }
        currentState = AppState.ANOMALY_DETECTED_ROOM_ONE

        status.text = "Anomaly Detected in room 1"
        status.visibility = View.VISIBLE
        time.text = data.getString("time")
        temp.text = data.getString("temperature")
        humidty.text = data.getString("humidity")
        voc.text = data.getString("voc")
        ir_light.text = data.getString("ir_light")
        visable_light.text = data.getString("visible_light")
        co2.text = data.getString("co2")
        stats_visable()

    }

    private fun anomaly_detected_room_2(data: JSONObject) {

        if (currentState!=AppState.ANOMALY_DETECTED_ROOM_ONE){
            status.text = "No Anomalies Found In Room 1"
            status.visibility = View.VISIBLE
        }
        currentState = AppState.ANOMALY_DETECTED_ROOM_TWO
        status2.text = "Anomaly Detected in room 2"

        time2.text = data.getString("time")
        temp2.text = data.getString("temperature")
        humidty2.text = data.getString("humidity")
        voc2.text = data.getString("voc")
        ir_light2.text = data.getString("ir_light")
        visable_light2.text = data.getString("visible_light")
        co2_2.text = data.getString("co2")
        time2.visibility = View.VISIBLE
        temp2.visibility = View.VISIBLE
        humidty2.visibility = View.VISIBLE
        voc2.visibility = View.VISIBLE
        ir_light2.visibility = View.VISIBLE
        visable_light2.visibility = View.VISIBLE
        co2_2.visibility = View.VISIBLE

        time_placeholder2.visibility = View.VISIBLE
        temp_placeholder2.visibility = View.VISIBLE
        humidty_placeholder2.visibility = View.VISIBLE
        voc_placeholder2.visibility = View.VISIBLE
        ir_light_placeholder2.visibility = View.VISIBLE
        visable_light_placeholder2.visibility = View.VISIBLE
        co2_placeholder2.visibility = View.VISIBLE

        resetButton.visibility = View.VISIBLE

    }


    private fun stats_visable(){
        time.visibility = View.VISIBLE
        temp.visibility = View.VISIBLE
        humidty.visibility = View.VISIBLE
        voc.visibility = View.VISIBLE
        ir_light.visibility = View.VISIBLE
        visable_light.visibility = View.VISIBLE
        co2.visibility = View.VISIBLE

        time_placeholder.visibility = View.VISIBLE
        temp_placeholder.visibility = View.VISIBLE
        humidty_placeholder.visibility = View.VISIBLE
        voc_placeholder.visibility = View.VISIBLE
        ir_light_placeholder.visibility = View.VISIBLE
        visable_light_placeholder.visibility = View.VISIBLE
        co2_placeholder.visibility = View.VISIBLE

        resetButton.visibility = View.VISIBLE
    }












}