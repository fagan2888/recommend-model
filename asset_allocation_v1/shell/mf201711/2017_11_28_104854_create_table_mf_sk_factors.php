<?php

use Illuminate\Database\Schema\Blueprint;
use Illuminate\Database\Migrations\Migration;

class CreateTableMfSkFactors extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        Schema::create('mf_sk_factors', function($table) {
            $table->increments('id');
            $table->string('factor_name');
            $table->text('factor_explain');
            $table->integer('factor_source')->unsigned()->comment('0:行情数据(无延迟), 1:财报数据(按财
报延迟)');
            $table->string('factor_kind');
	    $table->string('formula');
            $table->date('start_time')->default('0000-00-00')->comment('因子数据开始时间');
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
        Schema::drop('mf_sk_factors');
    }
}
