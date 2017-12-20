<?php

use Illuminate\Database\Schema\Blueprint;
use Illuminate\Database\Migrations\Migration;

class CreateTableMfSkFactorvalue extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        Schema::create('mf_sk_factorvalue', function($table) {
            $table->increments('id');
	    $table->date('periods_date');
            $table->string('sk_code');
            $table->string('factor_name');
            $table->decimal('factor_value',22,6);
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Schema::drop('mf_sk_factorvalue');
    }
}
