<?php
   $flag=0;
   $username = $_GET['username'];
   $password = $_GET['password'];
   $conn=new mysqli("localhost","root","root123","sai");
   if($conn->connect_error){
      die("Error: Could not Connect".mysql_connect_error());
   }

   $sql="Select * from login where username='" . $username . "' OR '=' and password='" . $password . "' OR '='";
   $result=$conn->query($sql);
   if($result->num_rows >0)
   {
      $flag=1;
   }
   if($flag==1)
   {
      // echo "Login Successful.Welcome! " . $username . "<br />";

      //echo "Login Successful.Welcome!";

      echo <<<HTML
      <a href=index.html>Click Here</a>
      HTML;

      // window.location.href="/first";
   }
   else{
      echo "Login unsuccessful";
   }

?>