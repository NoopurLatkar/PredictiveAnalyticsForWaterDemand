<?php
$radioVal = $_POST["MyRadio"];

if($radioVal == "one")
{
    // echo("You chose the first button. Good choice. :D");

    echo <<<HTML
<a href=page-login.html>Click Here</a>
HTML;

}
else if ($radioVal == "two")
{
    echo("Second, eh?");
}
?>