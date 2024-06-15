package com.coding.phototry

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import java.io.File


class MainActivity : AppCompatActivity() {

    private  lateinit var captureIV : ImageView
    private  lateinit var imageUrl : Uri
    private  lateinit var text : TextView

    private val contract = registerForActivityResult(ActivityResultContracts.TakePicture()){
        captureIV.setImageURI(null)
        captureIV.setImageURI(imageUrl)
        text.setText(imageUrl.toString())

    }

    private val open = registerForActivityResult(ActivityResultContracts.GetContent()){


    }



    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)


        imageUrl = createImageUri()
        captureIV = findViewById(R.id.captureImageView)
        text = findViewById(R.id.captureText)

        val captureImgBtn = findViewById<Button>(R.id.captureImgVBtn)
        captureImgBtn.setOnClickListener{
            contract.launch(imageUrl)
        }

        val selectImgBtn = findViewById<Button>(R.id.selectImgBtn)
        selectImgBtn.setOnClickListener{
            open.launch("image/*")
        }


        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }
        }

    private fun createImageUri():Uri {
        val image = File(filesDir, "camera_photos.png")
        return FileProvider.getUriForFile(this,
            "com.coding.phototry.FileProvider",
            image)

    }

    
}